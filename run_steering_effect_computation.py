import random
import argparse
import pickle
import matplotlib.pyplot as plt

from functools import partial

import requests
import random
import einops
from dotenv import load_dotenv
import os
import gc
import numpy as np
import pandas as pd
import pickle
import requests
import json
import torch as t
from sae_lens import (
    SAE,
    HookedSAETransformer,
)
from jaxtyping import Float
from tqdm.auto import tqdm
from torch import Tensor
from transformer_lens.hook_points import HookPoint

from run_upstream_feature_identification import (
    get_chat_template,
    load_model,
    load_refusal_dir,
    load_data,
    load_steer_targets,
)
from helper import get_coefficients, get_boundary_variant, get_valid_models

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)
print(f"using {device}")

VALID_MODELS = get_valid_models()
COMPARISON_LABEL = get_boundary_variant()


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
    return p.parse_args()


def get_max_act_approx(layer, feature_idx, model_name, neuronpedia_key):
    if model_name == "gemma2-2b":
        url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_idx}"
    elif model_name == "llama3.1-8b":
        url = f"https://www.neuronpedia.org/api/feature/llama3.1-8b/{layer}-llamascope-res-32k/{feature_idx}"
    elif model_name == "qwen2.5_7b":
        url = f"https://www.neuronpedia.org/api/feature/qwen2.5-7b-it/{layer}-resid-post-aa/{feature_idx}"
    else:
        raise NotImplementedError(f"{model_name} currently not supported")
    headers = {"X-Api-Key": neuronpedia_key}
    response = requests.get(url, headers=headers)
    return response.json()["maxActApprox"]


def compute_calibration_projections(
    model: HookedSAETransformer,
    harmful_df: pd.DataFrame,
    refusal_direction: Float[Tensor, "d_model"],
    refusal_layer: int,
    steer_targets: dict = None,
    model_name: str = None,
) -> dict:
    """
    Compute unsteered R for imperative (refused) and interrogative (not refused)
    prompts to establish calibration bounds [R_low, R_high].
    Returns dict with keys 'imperative' and 'interrogative', each a list of floats.
    """
    cal = {"refusal": [], "compliance": []}
    cols = list(steer_targets.keys())
    for i, row in harmful_df.iterrows():
        for c in cols:
            if i in steer_targets[c]:
                if c != "imperative":
                    outcome = "compliance"
                else:
                    outcome = "refusal"
            else:
                if c != "imperative":
                    outcome = "refusal"
                else:
                    outcome = "compliance"
            templated, _ = get_chat_template(row[c], model.tokenizer, model_name)
            proj = get_refusal_projection(
                model=model,
                prompt=templated,
                refusal_direction=refusal_direction,
                refusal_layer=refusal_layer,
            )
            cal[outcome].append(proj)

    return cal


def get_refusal_projection(
    model: HookedSAETransformer,
    prompt: str,
    refusal_direction: Float[Tensor, "d_model"],
    refusal_layer: int,
    upstream_direction: Float[Tensor, "d_model"] = None,
    upstream_layer: int = None,
    steering_coefficient: float = None,
) -> float:
    """
    Measure R = projection of residual stream onto the refusal direction
    at (refusal_layer, last prompt token), optionally with a steering intervention
    applied at upstream_layer.

    If upstream_direction / steering_coefficient are None, this is an unsteered
    baseline measurement.
    """
    refusal_direction_norm = refusal_direction / t.norm(refusal_direction, p=2)
    projection_store = []

    def projection_hook(activation, hook):
        act = activation[0, -1, :].float()
        proj = einops.einsum(
            refusal_direction_norm.float().to(act.device), act, "d, d ->"
        )
        projection_store.append(proj.detach().cpu().item())

    fwd_hooks = [(f"blocks.{refusal_layer}.hook_resid_pre", projection_hook)]

    if upstream_direction is not None and steering_coefficient is not None:
        _steer = partial(
            steering_hook,
            direction=upstream_direction,
            steering_coefficient=steering_coefficient,
            steer_prompt_only=True,
        )
        fwd_hooks.append((f"blocks.{upstream_layer}.hook_resid_post", _steer))

    model.reset_hooks()
    with t.no_grad():
        model.run_with_hooks(
            prompt,
            fwd_hooks=fwd_hooks,
            stop_at_layer=refusal_layer + 1,
            prepend_bos=False,
        )

    return projection_store[0]


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


# ── Main steering loop with R metric ─────────────────────────────────────────


def run_steering_with_refusal_metric(
    model: HookedSAETransformer,
    harmful_df: pd.DataFrame,
    steer_targets: dict[str, list[int]],
    steering_features: list[int],
    sae: SAE,
    refusal_direction: Float[Tensor, "d_model"],
    refusal_layer: int,
    upstream_layer: int,
    steering_factors: list[float],
    output_prefix: str,
    also_steer_refusal_dir: bool = True,
    mean_gradient_vec: Float[Tensor, "d_model"] = None,
    random_steering_features: list = [],
    model_name: str = None,
    neuronpedia_key: str = None,
) -> pd.DataFrame:
    """
    For each syntax form in steer_targets, steer the corresponding prompts
    with each feature × coefficient combination and record delta-R.

    steer_targets: {syntax_label: [prompt_indices]}
      - Non-imperative keys: prompts not yet refused → steered toward refusal
        (positive coefficients).
      - "imperative" key: prompts already refused → steered toward compliance
        (negative coefficients, i.e. steer_reverse=True for that form only).

    Returns a tidy DataFrame with one row per (syntax_label, pi, feature, coef).
    Positive delta_R means R increased (moved toward refusal) regardless of direction.
    """
    results_file = f"{output_prefix}_projection_results.pkl"
    if os.path.exists(results_file):
        print(f"Loading cached projection results from {results_file}")
        with open(results_file, "rb") as f:
            all_records = pickle.load(f)
    else:
        all_records = []

    already_done = {
        (r["syntax_label"], r["pi"], r["feature_idx"], r["coef"]) for r in all_records
    }

    for syntax_label, prompt_indices in steer_targets.items():
        # Imperative prompts are already refused → steer toward compliance (negative coef).
        # All other forms have prompts not yet refused → steer toward refusal (positive coef).
        steer_reverse = syntax_label == "imperative"
        prompts_col = harmful_df[syntax_label].tolist()
        for pi in tqdm(
            prompt_indices, desc=f"steering {syntax_label} (reverse={steer_reverse})"
        ):
            raw_prompt = prompts_col[pi]
            templated, _ = get_chat_template(raw_prompt, model.tokenizer, model_name)

            # ── Unsteered baseline R ──────────────────────────────────────────
            baseline_key = (syntax_label, pi, "baseline", 0)
            if baseline_key not in already_done:
                baseline_r = get_refusal_projection(
                    model=model,
                    prompt=templated,
                    refusal_direction=refusal_direction,
                    refusal_layer=refusal_layer,
                )
                all_records.append(
                    {
                        "syntax_label": syntax_label,
                        "pi": pi,
                        "feature_idx": "baseline",
                        "coef": 0,
                        "R": baseline_r,
                        "delta_R": 0.0,
                        "is_refusal_dir": False,
                        "is_refusal_grad": False,
                        "is_random_feature": False,
                    }
                )
                already_done.add(baseline_key)
            else:
                baseline_r = next(
                    r["R"]
                    for r in all_records
                    if (r["syntax_label"], r["pi"], r["feature_idx"], r["coef"])
                    == baseline_key
                )

            # ── Steer with refusal direction itself ───────────────────────────
            if also_steer_refusal_dir:
                sweep = compute_refusal_projection_sweep_refusal_dir(
                    model=model,
                    prompt=templated,
                    refusal_direction=refusal_direction,
                    refusal_layer=refusal_layer,
                    upstream_layer=upstream_layer,
                    steering_factors=steering_factors,
                    max_act_approx=30,  # just like in repo
                    steer_reverse=steer_reverse,
                )
                for coef, r_val in sweep.items():
                    key = (syntax_label, pi, "refusal_dir", coef)
                    if key not in already_done:
                        all_records.append(
                            {
                                "syntax_label": syntax_label,
                                "pi": pi,
                                "feature_idx": "refusal_dir",
                                "coef": coef,
                                "R": r_val,
                                "delta_R": r_val - baseline_r,
                                "is_refusal_dir": True,
                                "is_refusal_grad": False,
                                "is_random_feature": False,
                            }
                        )
                        already_done.add(key)

            # ── Steer with mean gradient ───────────────────────────
            if mean_gradient_vec is not None:
                sweep = compute_refusal_projection_sweep(
                    model=model,
                    prompt=templated,
                    upstream_direction=mean_gradient_vec,
                    upstream_layer=upstream_layer,
                    refusal_direction=refusal_direction,
                    refusal_layer=refusal_layer,
                    steering_factors=steering_factors,
                    max_act_approx=30,  # just like in repo
                    steer_reverse=steer_reverse,
                )
                for coef, r_val in sweep.items():
                    key = (syntax_label, pi, "mean_gradient", coef)
                    if key not in already_done:
                        all_records.append(
                            {
                                "syntax_label": syntax_label,
                                "pi": pi,
                                "feature_idx": "mean_gradient",
                                "coef": coef,
                                "R": r_val,
                                "delta_R": r_val - baseline_r,
                                "is_refusal_dir": False,
                                "is_refusal_grad": True,
                                "is_random_feature": False,
                            }
                        )
                        already_done.add(key)

            # ── Steer with each selected SAE feature ───────────────────────────────────
            for feature_idx in steering_features:
                try:
                    max_act = get_max_act_approx(
                        layer=upstream_layer,
                        feature_idx=feature_idx,
                        model_name=model_name,
                        neuronpedia_key=neuronpedia_key,
                    )
                except requests.exceptions.JSONDecodeError:
                    print(f"cannot process feature: {feature_idx} at L{upstream_layer}")
                    continue
                direction = sae.W_dec[feature_idx, :].detach()

                sweep = compute_refusal_projection_sweep(
                    model=model,
                    prompt=templated,
                    upstream_direction=direction,
                    upstream_layer=upstream_layer,
                    refusal_direction=refusal_direction,
                    refusal_layer=refusal_layer,
                    steering_factors=steering_factors,
                    max_act_approx=max_act,
                    steer_reverse=steer_reverse,
                )
                for coef, r_val in sweep.items():
                    key = (syntax_label, pi, feature_idx, coef)
                    if key not in already_done:
                        all_records.append(
                            {
                                "syntax_label": syntax_label,
                                "pi": pi,
                                "feature_idx": feature_idx,
                                "coef": coef,
                                "R": r_val,
                                "delta_R": r_val - baseline_r,
                                "is_refusal_dir": False,
                                "is_refusal_grad": False,
                                "is_random_feature": False,
                            }
                        )
                        already_done.add(key)

            # ── Steer with each random SAE feature ───────────────────────────────────
            for feature_idx in random_steering_features:
                max_act = get_max_act_approx(
                    layer=upstream_layer,
                    feature_idx=feature_idx,
                    model_name=model_name,
                    neuronpedia_key=neuronpedia_key,
                )
                direction = sae.W_dec[feature_idx, :].detach()

                sweep = compute_refusal_projection_sweep(
                    model=model,
                    prompt=templated,
                    upstream_direction=direction,
                    upstream_layer=upstream_layer,
                    refusal_direction=refusal_direction,
                    refusal_layer=refusal_layer,
                    steering_factors=steering_factors,
                    max_act_approx=max_act,
                    steer_reverse=steer_reverse,
                )
                for coef, r_val in sweep.items():
                    key = (syntax_label, pi, feature_idx, coef)
                    if key not in already_done:
                        all_records.append(
                            {
                                "syntax_label": syntax_label,
                                "pi": pi,
                                "feature_idx": feature_idx,
                                "coef": coef,
                                "R": r_val,
                                "delta_R": r_val - baseline_r,
                                "is_refusal_dir": False,
                                "is_refusal_grad": False,
                                "is_random_feature": True,
                            }
                        )
                        already_done.add(key)

        # Checkpoint after each syntax form
        with open(results_file, "wb") as f:
            pickle.dump(all_records, f)
        print(f"Saved checkpoint after syntax_label={syntax_label}")

    df = pd.DataFrame(all_records)
    return df


def compute_refusal_projection_sweep_refusal_dir(
    model: HookedSAETransformer,
    prompt: str,
    refusal_direction: Float[Tensor, "d_model"],
    refusal_layer: int,
    upstream_layer: int,
    steering_factors: list[float],
    max_act_approx: float,
    steer_reverse: bool = False,
) -> dict[float, float]:
    """
    Same sweep but steers with the refusal direction itself (not an SAE feature),
    scaled by raw coefficient (no max_act_approx needed).
    """
    factors = [c * (-1) if steer_reverse else c for c in steering_factors]
    if 0 not in factors:
        factors = [0] + factors

    results = {}
    for coef in factors:
        proj = get_refusal_projection(
            model=model,
            prompt=prompt,
            refusal_direction=refusal_direction,
            refusal_layer=refusal_layer,
            upstream_direction=refusal_direction,
            upstream_layer=upstream_layer,
            steering_coefficient=coef * max_act_approx if coef != 0 else 0.0,
        )
        results[coef] = proj
    return results


def compute_refusal_projection_sweep(
    model: HookedSAETransformer,
    prompt: str,
    upstream_direction: Float[Tensor, "d_model"],
    upstream_layer: int,
    refusal_direction: Float[Tensor, "d_model"],
    refusal_layer: int,
    steering_factors: list[float],
    max_act_approx: float,
    steer_reverse: bool = False,
) -> dict[float, float]:
    """
    Sweep steering coefficients and return {coef: R_projection} for each.
    Also returns coef=0 (unsteered baseline) regardless of sweep list.
    Mirrors explore_steering's sign convention for steer_reverse.
    """
    factors = [c * (-1) if steer_reverse else c for c in steering_factors]
    # Always include baseline
    if 0 not in factors:
        factors = [0] + factors

    results = {}
    for coef in factors:
        proj = get_refusal_projection(
            model=model,
            prompt=prompt,
            refusal_direction=refusal_direction,
            refusal_layer=refusal_layer,
            upstream_direction=upstream_direction,
            upstream_layer=upstream_layer,
            steering_coefficient=coef * max_act_approx if coef != 0 else 0.0,
        )
        results[coef] = proj
    return results


def summarise_projection_results(
    df: pd.DataFrame, calibration: dict = None
) -> pd.DataFrame:
    """
    Aggregate R and delta_R per (feature_idx, coef, syntax_label).
    Optionally annotate with normalised_R relative to calibration bounds.
    """
    agg = (
        df[df["feature_idx"] != "baseline"]
        .groupby(["syntax_label", "feature_idx", "coef"])
        .agg(
            mean_R=("R", "mean"),
            std_R=("R", "std"),
            mean_delta_R=("delta_R", "mean"),
            std_delta_R=("delta_R", "std"),
            n_prompts=("pi", "count"),
            is_random_feature=("is_random_feature", "first"),
        )
        .reset_index()
        .sort_values("mean_delta_R", ascending=False)
    )

    if calibration is not None:
        r_low = np.mean(calibration["compliance"])  # compliance distribution mean
        r_high = np.mean(calibration["refusal"])  # refusal distribution mean
        r_range = r_high - r_low if r_high != r_low else 1.0
        agg["normalised_R"] = (agg["mean_R"] - r_low) / r_range
        agg["normalised_delta_R"] = agg["mean_delta_R"] / r_range

    return agg


def identify_successful_features(summary_df, calibration):
    all_features = summary_df["feature_idx"].unique().tolist()
    r_low, r_high = float(np.mean(calibration["compliance"])), float(
        np.mean(calibration["refusal"])
    )
    steering_coefs = {}
    for feature in all_features:
        sub = summary_df[summary_df["feature_idx"] == feature].copy()
        if feature not in ["mean_gradient", "refusal_dir"]:
            sub_comparison = (
                sub[(sub["syntax_label"] == COMPARISON_LABEL) & (sub["coef"] > 0)]
                .copy()
                .reset_index(drop=True)
            )
            sub_imperative = (
                sub[(sub["syntax_label"] == "imperative") & (sub["coef"] < 0)]
                .copy()
                .reset_index(drop=True)
            )

            # TODO: also handle the case for negative features, were r_high and r_low are switched

            if len(sub_comparison[sub_comparison["mean_R"] > r_high]) > 0:
                refusal_coef = sub_comparison[sub_comparison["mean_R"] > r_low].loc[
                    lambda d: d["mean_R"].idxmax()
                ][
                    "coef"
                ]  # where R is maximal
            else:
                refusal_coef = None

            if len(sub_imperative[sub_imperative["mean_R"] < r_low]) > 0:
                compliance_coef = sub_imperative[sub_imperative["mean_R"] < r_low].loc[
                    lambda d: d["coef"].idxmax()
                ]["coef"]
            else:
                compliance_coef = None

            if refusal_coef and compliance_coef:
                print(f"adding coef for F{feature}")
                steering_coefs[feature] = {
                    "refusal_coef": refusal_coef,
                    "compliance_coef": compliance_coef,
                }
    return steering_coefs


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
        raise NotImplementedError(f"{model_name} currently not supported.").py

    load_dotenv()
    NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY")

    upstream_layer = args.upstream_layer

    SYNTAX_LABELS, PROMPT_VARIANTS, harmful_df = load_data()

    model, sae_model, cfg_dict, sparsity = load_model(
        model_name, upstream_layer, args.cache_dir
    )

    # load refusal dir
    (
        refusal_layer,
        refusal_pos,
        refusal_direction,
        refusal_direction_unit,
    ) = load_refusal_dir(model_name)

    random.seed(5)
    n_random = 10
    d_sae = sae_model.W_dec.shape[0]
    random_steering_features = random.sample(range(d_sae), n_random)

    with open(f"steering/{model_name}/steering_features.json") as f:
        all_steering_features = {
            int(l): features for l, features in json.load(f).items()
        }

    try:
        steering_features = all_steering_features[upstream_layer]
    except KeyError:
        print(f"Features for L{upstream_layer} not computed.")
        exit()

    # external syntax senstivity filter # with filtering (no activation diff considered) + top promotion + top suppression
    if model_name == "gemma2-2b" or model_name == "llama3.1-8b":
        refusal_pos = -1
    elif model_name == "qwen2.5_7b":
        refusal_pos = -4
    else:
        raise NotImplementedError

    cache_tag = f"l{upstream_layer}_pos{refusal_pos}"

    steering_factors = get_coefficients()

    STEERING_DIR = f"steering/{model_name}"
    SAE_CACHE_DIR = f"sae_cache/{model_name}"
    OUTPUT_PREFIX = f"{STEERING_DIR}/steering_l{upstream_layer}"

    steer_targets_file = f"{STEERING_DIR}/steering_targets.pkl"
    steer_targets = load_steer_targets(steer_targets_file)

    # ── Calibration ──────────────────────────────────────────────────────────
    cal_cache = f"sae_cache/{model_name}/calibration_{cache_tag}.pkl"
    if os.path.exists(cal_cache):
        with open(cal_cache, "rb") as f:
            calibration = pickle.load(f)
    else:
        calibration = compute_calibration_projections(
            model=model,
            harmful_df=harmful_df,
            refusal_direction=refusal_direction,
            refusal_layer=refusal_layer,
            steer_targets=steer_targets,
            model_name=model_name,
        )
        with open(cal_cache, "wb") as f:
            pickle.dump(calibration, f)
    print(
        f"Calibration R̄: compliance={np.mean(calibration['compliance']):.4f}  "
        f"refusal={np.mean(calibration['refusal']):.4f}"
    )

    # ── Run steering + measure R ──────────────────────────────────────────────
    mean_gradient_path = (
        f"{SAE_CACHE_DIR}/mean_{COMPARISON_LABEL}_gradient_vec_{cache_tag}.pt"
    )
    mean_gradient_vec = t.load(mean_gradient_path).to(device)

    df_projections = run_steering_with_refusal_metric(
        model=model,
        harmful_df=harmful_df,
        steer_targets=steer_targets,
        steering_features=steering_features,
        sae=sae_model,
        refusal_direction=refusal_direction,
        refusal_layer=refusal_layer,
        upstream_layer=upstream_layer,
        steering_factors=steering_factors,
        output_prefix=OUTPUT_PREFIX,
        also_steer_refusal_dir=True,
        mean_gradient_vec=mean_gradient_vec,
        random_steering_features=random_steering_features,
        model_name=model_name,
        neuronpedia_key=NEURONPEDIA_API_KEY,
    )

    # ── Summary ───────────────────────────────────────────────────────
    summary_df = summarise_projection_results(df_projections, calibration=calibration)
    summary_file = f"{OUTPUT_PREFIX}_projection_summary.csv"
    summary_df.to_csv(summary_file, sep=";", index=False)
    print(f"\nSummary saved to {summary_file}")
    print(summary_df.head(20).to_string(index=False))

    # ── Selection of steering coefficients for successful steering features
    steering_coefficients = identify_successful_features(summary_df, calibration)

    with open(f"{STEERING_DIR}/l{upstream_layer}_steering_config.pkl", "wb") as f:
        pickle.dump(steering_coefficients, f)
