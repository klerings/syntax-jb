import argparse
import os
import pickle

import einops
import numpy as np
import pandas as pd
import requests
import torch as t
from dotenv import load_dotenv
from sae_lens import SAE, HookedSAETransformer
from tqdm.auto import tqdm

from helper import get_boundary_variant, get_valid_models

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}.")

VALID_MODELS = get_valid_models()
COMPARISON_LABEL = get_boundary_variant()

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify upstream SAE features that influence refusal behaviour."
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=VALID_MODELS,
        help="Name of the model whose judge results are to be processed.",
    )
    parser.add_argument(
        "cache_dir",
        type=str,
        help="Directory to use as the HuggingFace model cache.",
    )
    parser.add_argument(
        "--upstream_layer",
        type=int,
        required=True,
        help="Upstream layer to search for refusal-influencing features.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Recompute all intermediate results even if cached files already exist.",
    )
    return parser.parse_args()


# ── Helper functions ──────────────────────────────────────────────────────────


def get_chat_template(prompt, tokenizer, model_name):
    if model_name == "gemma2-2b" or model_name == "qwen2.5_7b":
        prompt = [{"role": "user", "content": prompt}]
        prompt_with_template = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_with_template_tokens = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
    elif model_name == "llama3.1-8b":
        prompt_with_template = LLAMA3_CHAT_TEMPLATE.format(instruction=prompt)
        prompt_with_template_tokens = tokenizer.encode(
            prompt_with_template,
            add_special_tokens=False,
        )
    else:
        raise NotImplementedError(f"Chat template not implemented for {model_name}.")
    return prompt_with_template, prompt_with_template_tokens


def load_tensor(filename, device=device):
    if device == "mps":
        tensor = t.load(filename, map_location="cpu")
        tensor = tensor.to(device, dtype=t.float32)
    else:
        tensor = t.load(filename)
    return tensor


def load_data():
    harmful_df = pd.read_csv("data/jbb_harmful_variants.csv", delimiter=";")
    harmless_df = pd.read_csv("data/jbb_harmless_variants.csv", delimiter=";")
    syntax_labels = [c for c in harmful_df.columns]
    prompt_variants = harmful_df.to_dict(orient="records") + harmless_df.to_dict(
        orient="records"
    )
    return syntax_labels, prompt_variants, harmful_df


def load_steer_targets(path: str) -> dict[str, list[int]]:
    with open(path, "rb") as f:
        steer_targets = pickle.load(f)
    steer_targets_harmful = {}
    for variant, labels in steer_targets["harmful"].items():
        steer_targets_harmful[variant] = []
        for i, label in enumerate(labels):
            if variant != "imperative":
                if label == "complied":
                    steer_targets_harmful[variant].append(i)
            else:
                # For imperative, steer those prompts that are already refused.
                if label == "refusal":
                    steer_targets_harmful[variant].append(i)
    return steer_targets_harmful


def load_model(model_name, upstream_layer, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    if model_name == "gemma2-2b":
        model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
            "gemma-2-2b-it", device=device, cache_dir=cache_dir
        )
        sae_name = "gemma-scope-2b-pt-res-canonical"
        sae_id = f"layer_{upstream_layer}/width_16k/canonical"
    elif model_name == "llama3.1-8b":
        model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", device=device, cache_dir=cache_dir
        )
        sae_name = "llama_scope_lxr_8x"
        sae_id = f"l{upstream_layer}r_8x"
    elif model_name == "qwen2.5_7b":
        model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", device=device, cache_dir=cache_dir
        )
        sae_name = "qwen2.5-7b-instruct-andyrdt"
        sae_id = f"resid_post_layer_{upstream_layer}_trainer_1"
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")

    sae_model, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_name,
        sae_id=sae_id,
        device=str(device),
    )
    return model, sae_model, cfg_dict, sparsity


def load_refusal_dir(model_name):
    if model_name == "gemma2-2b":
        refusal_layer = 15
        refusal_pos = -1
        filename = (
            f"steering/gemma2-2b-it_direction_l{refusal_layer}_pos{refusal_pos}.pt"
        )
    elif model_name == "llama3.1-8b":
        refusal_layer = 12
        refusal_pos = -1
        filename = (
            f"steering/llama-3.1-8b-it_direction_l{refusal_layer}_pos{refusal_pos}.pt"
        )
    elif model_name == "qwen2.5_7b":
        refusal_layer = 19
        refusal_pos = -4
        filename = (
            f"steering/qwen2.5_7b_it_direction_l{refusal_layer}_pos{refusal_pos}.pt"
        )
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
    refusal_direction = load_tensor(filename, device=device)
    refusal_direction_unit = refusal_direction / t.norm(refusal_direction, p=2)
    return refusal_layer, refusal_pos, refusal_direction, refusal_direction_unit


def get_sae_activations(model, sae, prompt, layer, token_pos=-1):
    """Forward pass → residual stream at (layer, token_pos) → SAE encode. Returns a (d_sae,) tensor on CPU."""
    resid_store = []

    def cache_hook(activation, hook):
        resid_store.append(activation[0, token_pos, :].detach().cpu())

    model.reset_hooks()
    with t.no_grad():
        model.run_with_hooks(
            prompt,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", cache_hook)],
            stop_at_layer=layer + 1,
            prepend_bos=False,
        )

    resid = resid_store[0]
    sae_cpu = sae.to("cpu")
    sae_acts = sae_cpu.encode(resid.unsqueeze(0)).squeeze(0)
    sae.to(device)
    return sae_acts


def get_gradient(model, prompt, upstream_layer, direction, downstream_layer, pos):
    direction = direction / t.linalg.vector_norm(direction)
    model.reset_hooks()
    backward_cache = {}

    def upstream_gradient_hook(gradient, hook):
        backward_cache[hook.name] = gradient.detach()

    def downstream_projection_hook(activation, hook):
        d = direction.float().to(activation.device)
        projection = einops.einsum(
            d, activation.float(), "n_dim, batch ctx n_dim -> batch ctx"
        )[0, pos]
        projection.backward()

    bwd_hooks = [(f"blocks.{upstream_layer}.hook_resid_post", upstream_gradient_hook)]
    fwd_hooks = [
        (f"blocks.{downstream_layer}.hook_resid_pre", downstream_projection_hook)
    ]

    with t.enable_grad():
        _ = model.run_with_hooks(
            prompt,
            bwd_hooks=bwd_hooks,
            fwd_hooks=fwd_hooks,
            stop_at_layer=downstream_layer + 1,
            prepend_bos=False,
        )

    return backward_cache[f"blocks.{upstream_layer}.hook_resid_post"]


def compute_gradient_scores(
    model, sae, prompt, upstream_layer, refusal_direction, refusal_layer, pos
):
    gradient = get_gradient(
        model,
        prompt,
        upstream_layer=upstream_layer,
        direction=refusal_direction,
        downstream_layer=refusal_layer,
        pos=pos,
    )
    gradient_normalized = gradient / t.norm(gradient, p=2)

    projection = einops.einsum(
        sae.W_dec.detach(), gradient.squeeze(), "d_sae d_model, ctx d_model -> d_sae"
    ).cpu()

    cosine_sim = einops.einsum(
        sae.W_dec.detach(),
        gradient_normalized.squeeze(),
        "d_sae d_model, ctx d_model -> d_sae",
    ).cpu()

    return cosine_sim, projection, gradient


def get_sae_activations_full(model, sae, prompt, layer, mode="mean", max_len=None):
    """
    Forward pass → residual stream at all token positions → SAE encode.
    Returns (d_sae,) for mean/sum/max, or (seq_len, d_sae) for position mode.
    """
    resid_store = []

    def cache_hook(activation, hook):
        resid_store.append(activation[0].detach().cpu())

    model.reset_hooks()
    with t.no_grad():
        model.run_with_hooks(
            prompt,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", cache_hook)],
            stop_at_layer=layer + 1,
            prepend_bos=False,
        )

    resid = resid_store[0]
    sae_cpu = sae.to("cpu")
    sae_acts = sae_cpu.encode(resid)
    sae.to(device)

    if mode == "mean":
        return sae_acts.mean(dim=0)
    elif mode == "sum":
        return sae_acts.sum(dim=0)
    elif mode == "max":
        return sae_acts.max(dim=0).values
    elif mode == "position":
        if max_len is not None:
            seq_len, d_sae = sae_acts.shape
            if seq_len >= max_len:
                return sae_acts[:max_len]
            else:
                pad = t.zeros(max_len - seq_len, d_sae)
                return t.cat([sae_acts, pad], dim=0)
        return sae_acts
    else:
        raise ValueError(
            f"mode must be 'mean', 'sum', 'max', or 'position'. Got: {mode!r}"
        )


def aggregate_sae_over_prompts(model, sae, prompts, layer, mode="mean", max_len=None):
    """
    Run get_sae_activations_full over a list of prompts and stack results.
    Returns (num_prompts, d_sae) for mean/sum/max, or (num_prompts, max_len, d_sae) for position mode.
    """
    results = [
        get_sae_activations_full(model, sae, p, layer, mode=mode, max_len=max_len)
        for p in prompts
    ]
    return t.stack(results, dim=0)


def get_auto_interp(layer, feature_idx, model_name, neuronpedia_key):
    if model_name == "gemma2-2b":
        url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_idx}"
    elif model_name == "llama3.1-8b":
        url = f"https://www.neuronpedia.org/api/feature/llama3.1-8b/{layer}-llamascope-res-32k/{feature_idx}"
    elif model_name == "qwen2.5_7b":
        url = f"https://www.neuronpedia.org/api/feature/qwen2.5-7b-it/{layer}-resid-post-aa/{feature_idx}"
    else:
        raise NotImplementedError(f"{model_name} currently not supported.")
    headers = {"X-Api-Key": neuronpedia_key}
    response = requests.get(url, headers=headers)
    data = response.json()
    explanations = data.get("explanations", [])
    if len(explanations) == 0:
        print(f"No explanation for feature {feature_idx}.")
        return None
    return explanations[0].get("description", None)


def main():
    args = parse_args()

    load_dotenv()
    neuronpedia_api_key = os.getenv("NEURONPEDIA_API_KEY")

    model_name = args.model_name
    if model_name == "gemma2-2b-greedy":
        model_name = "gemma2-2b"
    elif model_name == "qwen2.5_7b_greedy":
        model_name = "qwen2.5_7b"
    elif model_name == "llama3.1-8b-greedy":
        model_name = "llama3.1-8b"
    else:
        raise NotImplementedError(f"{model_name} currently not supported.")

    upstream_layer = args.upstream_layer

    syntax_labels, prompt_variants, harmful_df = load_data()
    model, sae_model, cfg_dict, sparsity = load_model(
        model_name, upstream_layer, args.cache_dir
    )
    (
        refusal_layer,
        refusal_pos,
        refusal_direction,
        refusal_direction_unit,
    ) = load_refusal_dir(model_name)

    steer_targets_file = f"steering/{model_name}/steering_targets.pkl"
    steer_targets = load_steer_targets(steer_targets_file)

    sae_cache_dir = f"sae_cache/{model_name}"
    os.makedirs(sae_cache_dir, exist_ok=True)

    load_cached = not args.no_cache
    cache_tag = f"l{upstream_layer}_pos{refusal_pos}"
    d_sae = sae_model.W_dec.shape[0]

    # ── Compute average SAE feature activations per grammatical mood ──────────
    feature_act_across_moods_file = (
        f"{sae_cache_dir}/l{upstream_layer}_mean_sae_feature_acts_across_moods.pt"
    )
    if load_cached and os.path.exists(feature_act_across_moods_file):
        print(f"Loading cached mood activations: {feature_act_across_moods_file}")
        sae_mood_activations = t.load(feature_act_across_moods_file)
    else:
        print(f"Computing SAE activation differences for mood corpora.")
        label_to_acts = {}
        for label in ["imperative", "conditional", "interrogative"]:
            all_prompts = []
            max_len = 0
            with open(f"data/grammatical_mood_corpora/labelled_{label}.txt", "r") as f:
                lines = [line.strip() for line in f.readlines()]
            for sent in tqdm(lines, desc=f"Collecting {label}"):
                templated, templated_tokens = get_chat_template(
                    sent, model.tokenizer, model_name
                )
                all_prompts.append(templated)
                max_len = max(max_len, len(templated_tokens))
            label_to_acts[label] = aggregate_sae_over_prompts(
                model=model,
                sae=sae_model,
                prompts=all_prompts,
                layer=upstream_layer,
                mode="mean",
                max_len=max_len,
            )

        sae_mood_activations = {
            "mean_diff_imp-cond": (
                label_to_acts["imperative"].mean(dim=0)
                - label_to_acts["conditional"].mean(dim=0)
            ).cpu(),
            "mean_diff_imp-interrog": (
                label_to_acts["imperative"].mean(dim=0)
                - label_to_acts["interrogative"].mean(dim=0)
            ).cpu(),
            "mean_diff_imp-cond_interrog": (
                label_to_acts["imperative"].mean(dim=0)
                - t.cat(
                    [label_to_acts["conditional"], label_to_acts["interrogative"]]
                ).mean(dim=0)
            ).cpu(),
        }
        t.save(sae_mood_activations, feature_act_across_moods_file)

    # ── Build base dataframe of SAE features ranked by mood activation diffs ──
    data_for_df = {"sae_feature": t.arange(d_sae).numpy()}
    for label, score in sae_mood_activations.items():
        data_for_df[label] = score.detach().cpu()
    df = pd.DataFrame(data_for_df)
    print("Ranked SAE features by activation differences.")

    # ── Compute gradient scores and identify top features per variant ─────────
    boundary_indices = steer_targets.get(COMPARISON_LABEL, [])
    if not boundary_indices:
        raise ValueError(
            f"{COMPARISON_LABEL} key not found in steer_targets pkl. "
            f"The gradient loop requires boundary prompts for {COMPARISON_LABEL}."
        )

    hf_prompts = harmful_df[COMPARISON_LABEL].tolist()
    mean_cs_path = f"{sae_cache_dir}/mean_{COMPARISON_LABEL}_cs_{cache_tag}.pt"
    mean_gradient_path = (
        f"{sae_cache_dir}/mean_{COMPARISON_LABEL}_gradient_vec_{cache_tag}.pt"
    )

    if load_cached and all(
        os.path.exists(p) for p in [mean_cs_path, mean_gradient_path]
    ):
        print(
            f"Loading cached gradients for layer {upstream_layer}, label {COMPARISON_LABEL}."
        )
        mean_gradient_vec = t.load(mean_gradient_path).to(device)
        mean_cosine_sim = t.load(mean_cs_path).to(device)
    else:
        for i in tqdm(boundary_indices, desc="Computing gradients"):
            outdir_cosine = (
                f"{sae_cache_dir}/cosine_sim_{COMPARISON_LABEL}_{cache_tag}_p{i}.pt"
            )
            outdir_gradient = (
                f"{sae_cache_dir}/raw_grad_{COMPARISON_LABEL}_{cache_tag}_p{i}.pt"
            )

            if load_cached and all(
                os.path.exists(p) for p in [outdir_cosine, outdir_gradient]
            ):
                print(f"Skipping gradient (cached): prompt {i}.")
                continue

            templated_prompt, _ = get_chat_template(
                hf_prompts[i], model.tokenizer, model_name
            )
            cosine_sim, projection, gradient = compute_gradient_scores(
                model,
                sae_model,
                templated_prompt,
                upstream_layer,
                refusal_direction,
                refusal_layer,
                refusal_pos,
            )
            t.save(cosine_sim, outdir_cosine)
            t.save(gradient.detach().cpu(), outdir_gradient)

        cosine_files = [
            f"{sae_cache_dir}/cosine_sim_{COMPARISON_LABEL}_{cache_tag}_p{i}.pt"
            for i in boundary_indices
            if os.path.exists(
                f"{sae_cache_dir}/cosine_sim_{COMPARISON_LABEL}_{cache_tag}_p{i}.pt"
            )
        ]
        all_cosine_sim = t.stack([t.load(f) for f in cosine_files])
        mean_cosine_sim = all_cosine_sim.mean(dim=0)
        t.save(mean_cosine_sim, mean_cs_path)

        grad_files = [
            f"{sae_cache_dir}/raw_grad_{COMPARISON_LABEL}_{cache_tag}_p{i}.pt"
            for i in boundary_indices
            if os.path.exists(
                f"{sae_cache_dir}/raw_grad_{COMPARISON_LABEL}_{cache_tag}_p{i}.pt"
            )
        ]
        all_grads = t.stack([t.load(f).squeeze()[-1] for f in grad_files])
        mean_gradient_vec = all_grads.mean(dim=0).to(device)
        t.save(mean_gradient_vec, mean_gradient_path)

    df_mean = pd.DataFrame(
        {
            "sae_feature": range(d_sae),
            "gradient_score": mean_cosine_sim.tolist(),
        }
    )
    df_merged = df.merge(df_mean, on="sae_feature")
    print("Merged SAE activation differences with upstream refusal relevance.")

    if COMPARISON_LABEL == "conditional":
        diff_types = ["mean_diff_imp-cond", "mean_diff_imp-cond_interrog"]
    elif COMPARISON_LABEL in ["present_active", "past_active"]:
        diff_types = ["mean_diff_imp-interrog", "mean_diff_imp-cond_interrog"]
    else:
        raise NotImplementedError(
            f"diff_types not defined for label: {COMPARISON_LABEL}"
        )

    for diff_type in diff_types:
        df_promotion = df_merged[df_merged[diff_type] > 0].copy()
        df_suppression = df_merged[df_merged[diff_type] < 0].copy()

        top_promotion = df_promotion.nlargest(50, "gradient_score")
        top_suppression = df_suppression.nsmallest(50, "gradient_score")

        top_promotion["interpretation"] = top_promotion["sae_feature"].apply(
            lambda idx: get_auto_interp(
                upstream_layer, idx, model_name, neuronpedia_api_key
            )
        )
        top_promotion.to_csv(
            f"{sae_cache_dir}/top50_promotion_features_{upstream_layer}_grad_of_{COMPARISON_LABEL}_filtered_by_{diff_type}_mean_over_pos.csv",
            index=False,
        )

        top_suppression["interpretation"] = top_suppression["sae_feature"].apply(
            lambda idx: get_auto_interp(
                upstream_layer, idx, model_name, neuronpedia_api_key
            )
        )
        top_suppression.to_csv(
            f"{sae_cache_dir}/top50_suppression_features_{upstream_layer}_grad_of_{COMPARISON_LABEL}_filtered_by_{diff_type}_mean_over_pos.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
