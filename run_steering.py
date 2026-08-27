import os
import sys
import pickle
import gc
import argparse

import numpy as np
import pandas as pd
import torch as t
from functools import partial
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from wildguard import load_wildguard
from collections import defaultdict
import random

from sae_lens import HookedSAETransformer
from transformer_lens.hook_points import HookPoint


from run_upstream_feature_identification import (
    load_steer_targets,
    load_model,
    load_refusal_dir,
    get_chat_template,
)
from run_steering_effect_computation import (
    get_max_act_approx,
    compute_calibration_projections,
)
from helper import get_variants, get_boundary_variant, get_valid_models

VALID_MODELS = get_valid_models()
VALID_VARIANTS = get_variants()
COMPARISON_LABEL = get_boundary_variant()

device = "cuda" if t.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "model_name",
        type=str,
        choices=VALID_MODELS,
        help="Name of the model whose judge results are to be processed.",
    )
    p.add_argument(
        "cache_dir",
        type=str,
        help="Directory to use as the HuggingFace model cache.",
    )
    p.add_argument(
        "--upstream_layer",
        type=int,
        required=True,
        help="Upstream layer to search for refusal influencing features.",
    )
    p.add_argument(
        "--random", action="store_true", help="Whether to use random feature baseline"
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_prompts(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, delimiter=";")


def steering_hook(
    activations: Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    direction: Float[Tensor, "d_model"],
    steering_coefficient: float = None,
    steer_prompt_only: bool = True,
) -> Tensor:
    direction = direction / t.norm(direction, p=2)
    if activations.shape[1] != 1:
        activations[:, :, :] = activations[:, :, :] + steering_coefficient * direction
    else:
        if not steer_prompt_only:
            activations[:, :, :] = (
                activations[:, :, :] + steering_coefficient * direction
            )
    return activations


def generate_with_steering(
    model: HookedSAETransformer,
    upstream_layer: int,
    prompt: str,
    direction: Float[Tensor, "d_model"],
    steering_coefficient: float = None,
    max_new_tokens: int = 50,
    return_response_only: bool = True,
    steer_prompt_only: bool = True,
    model_name: str = None,
):
    model.reset_hooks()
    _steering_hook = partial(
        steering_hook,
        direction=direction,
        steering_coefficient=steering_coefficient,
        steer_prompt_only=steer_prompt_only,
    )
    templated_prompt, _ = get_chat_template(prompt, model.tokenizer, model_name)
    with model.hooks(
        fwd_hooks=[(f"blocks.{upstream_layer}.hook_resid_post", _steering_hook)]
    ):
        output = model.generate(
            templated_prompt,
            max_new_tokens=max_new_tokens,
            prepend_bos=False,
            do_sample=False,
            verbose=False,
        )
    # this doesn't split correctly for all models, for qwen, for instance, it is not "model" but "assistant"
    if return_response_only:
        if model_name == "gemma2-2b":
            if "model\n" in output:
                return output.split("model\n", 1)[1]
            return output
        elif model_name == "qwen2.5_7b":
            if "assistant\n" in output:
                return output.split("model\n", 1)[1]
            return output
        else:
            return output
    return output


def generate_all(
    model,
    sae,
    feature,
    upstream_layer,
    refusal_coef,
    compliance_coef,
    steer_targets,
    harmful_df,
    harmless_df,
    model_name,
    device,
    max_feature_act=None,
):
    results = []

    for prompt_mode, prompt_df in zip(
        ["harmful", "harmless"], [harmful_df, harmless_df]
    ):
        for pi, row in tqdm(prompt_df.iterrows(), desc=f"generating for {prompt_mode}"):
            for syntax_label in VALID_VARIANTS:
                prompt = row[syntax_label]

                # ── direction selection ─────────────────────────────
                if feature == "refusal_dir":
                    # load refusal direction
                    _, direction, _ = load_refusal_dir()
                    max_act = 30

                elif feature == "mean_gradient":
                    # load mean gradient vec
                    direction = t.load(
                        f"sae_cache/{model_name}/mean_{COMPARISON_LABEL}_gradient_vec_l{upstream_layer}_pos-1.pt",
                        map_location=device,
                    )
                    max_act = 30

                else:
                    direction = sae.W_dec[feature, :].detach()
                    max_act = max_feature_act

                original_label = steer_targets[prompt_mode][syntax_label][pi]

                if original_label == "complied":
                    if syntax_label == "imperative":
                        continue
                    effective_coef = refusal_coef
                else:
                    if syntax_label != "imperative":
                        continue
                    effective_coef = compliance_coef

                response = generate_with_steering(
                    model=model,
                    upstream_layer=upstream_layer,
                    prompt=prompt,
                    direction=direction,
                    steering_coefficient=effective_coef * max_act,
                    max_new_tokens=300,
                )

                results.append(
                    {
                        "upstream_layer": upstream_layer,
                        "feature_idx": feature,
                        "coef": effective_coef,
                        "syntax_label": syntax_label,
                        "pi": pi,
                        "prompt_mode": prompt_mode,
                        "original_label": original_label,
                        "prompt": prompt,
                        "response": response,
                    }
                )
        print(f"finished: {prompt_mode}")

    return results


# ─────────────────────────────────────────────────────────────
# WILDGUARD PHASE
# ─────────────────────────────────────────────────────────────


def classify_batch(wildguard, items, batch_size=32):
    out = []
    for i in tqdm(range(0, len(items), batch_size), desc="WildGuard"):
        batch = items[i : i + batch_size]
        out.extend(wildguard.classify(batch))
    return out


def classify_results(wildguard, gen_results):
    items = [{"prompt": r["prompt"], "response": r["response"]} for r in gen_results]

    wg = classify_batch(wildguard, items)

    final = []
    for r, w in zip(gen_results, wg):
        final.append(
            {
                **r,
                "prompt_harmfulness": w["prompt_harmfulness"],
                "response_harmfulness": w["response_harmfulness"],
                "response_refusal": w["response_refusal"],
            }
        )

    return final


if __name__ == "__main__":
    # parse arguments
    args = parse_args()

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

    steer_targets_file = f"steering/{model_name}/steering_targets.pkl"
    steer_targets = load_steer_targets(steer_targets_file)

    print("Loading Model...")
    model, sae_model, _, _ = load_model(model_name, upstream_layer, args.cache_dir)

    steering_config_path = (
        f"steering/{model_name}/l{upstream_layer}_steering_config.pkl"
    )
    steering_config = load_config(steering_config_path)

    # baseline: use random SAE features
    if args.random:
        random_suffix = "random_"
        random.seed(5)
        n_random = 3
        d_sae = sae_model.W_dec.shape[0]
        # use random features that don't overlap with actual steering features
        found_random = False
        while not found_random:
            random_steering_features = random.sample(range(d_sae), n_random)
            if not any(x in steering_config for x in random_steering_features):
                found_random = True
                # overwrite steering config with new steering features
                # reuse same coefficients as for best feature
                steering_config = {
                    f: {"refusal_coef": 1.5, "compliance_coef": -2.5}
                    for f in random_steering_features
                }
    else:
        random_suffix = ""

    OUT_DIR_GEN = f"steering/{model_name}/generations"
    OUT_DIR_EVAL = f"steering/{model_name}/wildguard_eval"
    os.makedirs(OUT_DIR_GEN, exist_ok=True)
    os.makedirs(OUT_DIR_EVAL, exist_ok=True)

    harmful_df = load_prompts("data/jbb_harmful_variants.csv")
    harmless_df = load_prompts("data/jbb_harmless_variants.csv")

    generated_results = []

    for feature, coef_dict in tqdm(steering_config.items()):
        refusal_coef = coef_dict["refusal_coef"]
        compliance_coef = coef_dict["compliance_coef"]

        file_suffix = f"l{upstream_layer}_{random_suffix}feature={feature}_ref_coef={refusal_coef}_com_coef={compliance_coef}"

        gen_cache_path = os.path.join(OUT_DIR_GEN, f"{file_suffix}.pkl")
        if os.path.exists(gen_cache_path):
            print(f"Loading cached gen_results from {gen_cache_path}")
            with open(gen_cache_path, "rb") as f:
                gen_results = pickle.load(f)
        else:
            direction = sae_model.W_dec[feature, :].detach()
            max_feature_act = get_max_act_approx(upstream_layer, feature)

            # ─────────────────────────────
            # GENERATE
            # ─────────────────────────────
            gen_results = generate_all(
                model=model,
                sae=sae_model,
                feature=feature,
                upstream_layer=upstream_layer,
                refusal_coef=refusal_coef,
                compliance_coef=compliance_coef,
                steer_targets=steer_targets,
                harmful_df=harmful_df,
                harmless_df=harmless_df,
                model_name=model_name,
                device=device,
                max_feature_act=max_feature_act,
            )

            with open(gen_cache_path, "wb") as f:
                pickle.dump(gen_results, f)
            print(f"Saved gen_results: {gen_cache_path}")

        generated_results.append((gen_results, file_suffix))

    # ─────────────────────────────
    # FREE Model
    # ─────────────────────────────
    print("Freeing Model...")
    del model
    del sae_model
    # del refusal_direction
    # del mean_gradient_vec
    t.cuda.empty_cache()
    gc.collect()
    t.cuda.empty_cache()

    # ─────────────────────────────
    # LOAD WILDGUARD
    # ─────────────────────────────
    print("Loading WildGuard...")
    wildguard = load_wildguard()

    for (gen_results, file_suffix) in generated_results:

        out_path = os.path.join(OUT_DIR_EVAL, f"all_classified_{file_suffix}.csv")

        if all(os.path.exists(p) for p in [out_path]):
            continue

        final = classify_results(wildguard, gen_results)

        df = pd.DataFrame(final)
        df.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")
