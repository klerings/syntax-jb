import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from helper import get_variants


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANTS = get_variants()
N_SAMPLES = 100  # total samples per class per variant


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_resids(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_resid_layerwise(out_dir: str, variant: str) -> dict[str, dict]:
    """
    Returns dict: {"harmful": {layer: tensor}, "harmless": {layer: tensor}}
    Each tensor has shape (N_SAMPLES, hidden_dim) in float32.
    """
    both = {}
    for request_type in ["harmful", "harmless"]:
        path = os.path.join(out_dir, f"{request_type}_{variant}.pt")
        resid = load_resids(path)
        both[request_type] = {l: r.cpu().to(torch.float32) for l, r in resid.items()}
    return both


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def split_variant(X_dict: dict, test_indices: list) -> dict:
    """
    Splits each class tensor into train/test based on test_indices.

    Returns:
        {
          "harmful":  (X_train_dict, X_test_dict),
          "harmless": (X_train_dict, X_test_dict),
        }
    """
    all_indices = torch.arange(N_SAMPLES)
    test_mask = torch.isin(all_indices, torch.tensor(test_indices))

    out = {}
    for request_type, layer_dict in X_dict.items():
        X_train, X_test = {}, {}
        for layer_idx, resid in layer_dict.items():
            X_train[layer_idx] = resid[~test_mask]
            X_test[layer_idx] = resid[test_mask]
        out[request_type] = (X_train, X_test)
    return out


# ---------------------------------------------------------------------------
# K-fold CV split helpers
# ---------------------------------------------------------------------------


def make_kfold_indices(n_samples: int, n_folds: int, seed: int) -> list[tuple]:
    """
    Returns a list of (train_indices, test_indices) tuples, one per fold.
    Indices are shuffled once then partitioned into n_folds contiguous blocks.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples).tolist()
    fold_size = n_samples // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        # Last fold absorbs any remainder
        end = (k + 1) * fold_size if k < n_folds - 1 else n_samples
        test_idx = idx[start:end]
        train_idx = idx[:start] + idx[end:]
        folds.append((train_idx, test_idx))
    return folds


def split_by_indices(X_dict: dict, train_indices: list, test_indices: list) -> dict:
    """Generic split given explicit train and test index lists."""
    all_indices = torch.arange(N_SAMPLES)
    test_mask = torch.isin(all_indices, torch.tensor(test_indices))
    out = {}
    for request_type, layer_dict in X_dict.items():
        X_train, X_test = {}, {}
        for layer_idx, resid in layer_dict.items():
            X_train[layer_idx] = resid[~test_mask]
            X_test[layer_idx] = resid[test_mask]
        out[request_type] = (X_train, X_test)
    return out


def average_layerwise(fold_results: list[dict]) -> tuple[dict, dict]:
    """
    Given a list of {layer: acc} dicts (one per fold), return
    (mean_per_layer, std_per_layer).
    """
    layers = fold_results[0].keys()
    mean_lw, std_lw = {}, {}
    for layer in layers:
        vals = [f[layer] for f in fold_results]
        mean_lw[layer] = round(float(np.mean(vals)), 4)
        std_lw[layer] = round(float(np.std(vals)), 4)
    return mean_lw, std_lw


# ---------------------------------------------------------------------------
# Tensor combining helpers
# ---------------------------------------------------------------------------


def combine_hf_hl(X_hf: dict, X_hl: dict):
    """Concatenate harmful and harmless tensors; return (X_combined, labels)."""
    X = {}
    for layer_idx in X_hf:
        X[layer_idx] = torch.cat([X_hf[layer_idx], X_hl[layer_idx]], dim=0)
    n_hf = X_hf[next(iter(X_hf))].shape[0]
    n_hl = X_hl[next(iter(X_hl))].shape[0]
    y = [1] * n_hf + [0] * n_hl
    return X, y


# ---------------------------------------------------------------------------
# Mean-centering
# ---------------------------------------------------------------------------


def mean_center(train_hf: dict, train_hl: dict, test: dict):
    """
    Subtracts the combined training mean (harmful + harmless) from train and test.
    Returns centered_train_hf, centered_train_hl, centered_test.
    """
    c_hf, c_hl, c_test = {}, {}, {}
    for layer in train_hf:
        combined = torch.cat([train_hf[layer], train_hl[layer]], dim=0)
        mu = combined.mean(dim=0, keepdim=True)
        c_hf[layer] = train_hf[layer] - mu
        c_hl[layer] = train_hl[layer] - mu
        c_test[layer] = test[layer] - mu
    return c_hf, c_hl, c_test


# ---------------------------------------------------------------------------
# Diff-in-means probe
# ---------------------------------------------------------------------------


def extract_direction(train_hf: dict, train_hl: dict) -> dict:
    """Compute (mean_harmful - mean_harmless) direction per layer."""
    dirs = {}
    for layer in train_hf:
        dirs[layer] = train_hf[layer].mean(0) - train_hl[layer].mean(0)
    return dirs


def predict_dim(dirs: dict, X_test: dict, pred_mode: str) -> dict:
    """
    Apply the diff-in-means direction to test activations.

    pred_mode:
        "hard"  -> 1 if logit > 0, 0 otherwise  (matches mean-centered decision boundary)
        "soft"  -> sigmoid(logit) > 0.5
    """
    preds_per_layer = {}
    for layer, direction in dirs.items():
        logits = X_test[layer] @ direction
        if pred_mode == "hard":
            preds = (logits > 0).float()
        else:
            preds = torch.sigmoid(logits)
        preds_per_layer[layer] = preds.cpu()
    return preds_per_layer


def compute_layerwise_acc(preds_layerwise: dict, labels: list) -> dict:
    acc = {}
    labels_tensor = torch.tensor(labels, device="cpu")
    for layer, preds in preds_layerwise.items():
        binary = (preds > 0.5).long()
        score = (binary == labels_tensor).float().mean().item()
        acc[layer] = round(score, 4)
    return acc


# ---------------------------------------------------------------------------
# Cluster probe
# ---------------------------------------------------------------------------


def safe_cosine_sim(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)
    return torch.dot(a, b) / (norm_a * norm_b + eps)


def extract_centroids(train_hf: dict, train_hl: dict) -> dict:
    return {
        layer: (train_hf[layer].mean(dim=0), train_hl[layer].mean(dim=0))
        for layer in train_hf
    }


def cluster_probe(centroids: dict, X_test: dict, labels: list) -> float:
    """
    Predict class by comparing cosine similarity to harmful vs harmless centroids,
    averaged across all layers.
    """
    layers = sorted(centroids.keys())
    num_samples = len(labels)

    preds = []
    for sample_id in range(num_samples):
        layer_scores = []
        for layer in layers:
            h = X_test[layer][sample_id]
            c_hf, c_hl = centroids[layer]
            sim_hf = safe_cosine_sim(h, c_hf)
            sim_hl = safe_cosine_sim(h, c_hl)
            layer_scores.append((sim_hf - sim_hl).item())

        s_mean = np.nanmean(layer_scores)
        preds.append(1 if s_mean >= 0 else 0)

    return float(np.mean(np.array(labels) == np.array(preds)))


# ---------------------------------------------------------------------------
# Main probe routine
# ---------------------------------------------------------------------------


def run_probe(
    train_hf: dict, train_hl: dict, X_test: dict, y_test: list, pred_mode: str
):
    """
    Mean-center, then run both probes. Returns (acc_layerwise, acc_cluster).
    """
    c_hf, c_hl, c_test = mean_center(train_hf, train_hl, X_test)

    dirs = extract_direction(c_hf, c_hl)
    preds_dim = predict_dim(dirs, c_test, pred_mode)
    acc_lw = compute_layerwise_acc(preds_dim, y_test)

    centroids = extract_centroids(c_hf, c_hl)
    acc_cl = cluster_probe(centroids, c_test, y_test)

    return acc_lw, acc_cl


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--position",
        type=str,
        default="mean",
        help="Token position (e.g. last_prompt, final, mean)",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="Prediction mode for diff-in-means: hard or soft",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None, help="Model names to run."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.3,
        help="Fraction of samples to use as test set (single-split mode).",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use k-fold cross-validation instead of a single random split.",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds for CV (default: 5)."
    )
    args = parser.parse_args()

    model_names = args.models

    # Build fold definitions once — same folds used for every model and variant
    if args.cv:
        folds = make_kfold_indices(N_SAMPLES, args.n_folds, args.seed)
        print(
            f"Running {args.n_folds}-fold CV  "
            f"(~{N_SAMPLES - len(folds[0][1])} train / ~{len(folds[0][1])} test per fold)"
        )
    else:
        random.seed(args.seed)
        n_test = int(N_SAMPLES * args.test_frac)
        test_indices = random.sample(range(N_SAMPLES), n_test)
        folds = [(list(set(range(N_SAMPLES)) - set(test_indices)), test_indices)]

    for model_name in model_names:
        out_dir = f"probing/resids_{args.position}/{model_name}"

        if not os.path.isdir(out_dir):
            print(f"Skipping {model_name}: directory not found.")
            continue

        print(f"\n=== {model_name} | pos={args.position} | pred={args.pred} ===")

        # Load all variants upfront (no split yet — splitting happens per fold)
        all_variants = {}
        for variant in VARIANTS:
            all_variants[variant] = load_resid_layerwise(out_dir, variant)

        # Accumulators: per (train_v, test_v), collect results across folds
        # fold_lw_results[train_v][test_v] = list of {layer: acc}  (one per fold)
        # fold_cl_results[train_v][test_v] = list of float          (one per fold)
        fold_lw_results = {tv: {tv2: [] for tv2 in VARIANTS} for tv in VARIANTS}
        fold_cl_results = {tv: {tv2: [] for tv2 in VARIANTS} for tv in VARIANTS}

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if args.cv:
                print(
                    f"  Fold {fold_idx + 1}/{args.n_folds} "
                    f"(train={len(train_idx)}, test={len(test_idx)})"
                )

            # Split every variant for this fold
            variant_splits = {
                v: split_by_indices(all_variants[v], train_idx, test_idx)
                for v in VARIANTS
            }

            for train_variant in VARIANTS:
                train_hf = variant_splits[train_variant]["harmful"][0]
                train_hl = variant_splits[train_variant]["harmless"][0]

                for test_variant in VARIANTS:
                    test_hf = variant_splits[test_variant]["harmful"][1]
                    test_hl = variant_splits[test_variant]["harmless"][1]
                    X_test, y_test = combine_hf_hl(test_hf, test_hl)

                    acc_lw, acc_cl = run_probe(
                        train_hf, train_hl, X_test, y_test, args.pred
                    )

                    fold_lw_results[train_variant][test_variant].append(acc_lw)
                    fold_cl_results[train_variant][test_variant].append(acc_cl)

        # ----------------------------------------------------------------
        # Aggregate across folds
        # ----------------------------------------------------------------
        # acc_lw_results[train_v][test_v] = {layer: mean_acc}
        # acc_lw_std    [train_v][test_v] = {layer: std_acc}
        # acc_cl_results[train_v][test_v] = mean_float
        # acc_cl_std    [train_v][test_v] = std_float
        acc_lw_results = {}
        acc_lw_std = {}
        acc_cl_results = {}
        acc_cl_std = {}

        for train_v in VARIANTS:
            acc_lw_results[train_v] = {}
            acc_lw_std[train_v] = {}
            acc_cl_results[train_v] = {}
            acc_cl_std[train_v] = {}
            for test_v in VARIANTS:
                mean_lw, std_lw = average_layerwise(fold_lw_results[train_v][test_v])
                acc_lw_results[train_v][test_v] = mean_lw
                acc_lw_std[train_v][test_v] = std_lw

                cl_vals = fold_cl_results[train_v][test_v]
                acc_cl_results[train_v][test_v] = round(float(np.mean(cl_vals)), 4)
                acc_cl_std[train_v][test_v] = round(float(np.std(cl_vals)), 4)

                print(
                    f"  train={train_v:20s}  test={test_v:20s}  "
                    f"cluster_acc={acc_cl_results[train_v][test_v]:.4f}"
                    + (f" ± {acc_cl_std[train_v][test_v]:.4f}" if args.cv else "")
                )

        # ----------------------------------------------------------------
        # Save results
        # ----------------------------------------------------------------
        suffix = f"cv{args.n_folds}" if args.cv else "split"
        probe_dir = f"probing/variant_probes_{args.position}_{args.pred}_{suffix}"
        os.makedirs(probe_dir, exist_ok=True)

        # 1. Full layerwise detail as pickle
        #    Includes mean, std, and raw per-fold results for further analysis
        layerwise_payload = {
            "mean": acc_lw_results,
            "std": acc_lw_std,
            "per_fold": fold_lw_results,  # list of {layer: acc} per (train_v, test_v)
        }
        with open(f"{probe_dir}/{model_name}_layerwise.pkl", "wb") as f:
            pickle.dump(layerwise_payload, f)

        # 2. Cluster probe heatmap CSV (mean)
        heatmap_cl = pd.DataFrame(
            {
                test_v: {
                    train_v: acc_cl_results[train_v][test_v] for train_v in VARIANTS
                }
                for test_v in VARIANTS
            }
        )
        heatmap_cl.to_csv(f"{probe_dir}/{model_name}_cluster_heatmap.csv")

        # 3. Cluster probe std CSV (only meaningful for CV)
        if args.cv:
            heatmap_cl_std = pd.DataFrame(
                {
                    test_v: {
                        train_v: acc_cl_std[train_v][test_v] for train_v in VARIANTS
                    }
                    for test_v in VARIANTS
                }
            )
            heatmap_cl_std.to_csv(f"{probe_dir}/{model_name}_cluster_heatmap_std.csv")

        # 4. Diff-in-means best-layer heatmap CSV (max mean acc over layers)
        heatmap_dim = pd.DataFrame(
            {
                test_v: {
                    train_v: max(acc_lw_results[train_v][test_v].values())
                    for train_v in VARIANTS
                }
                for test_v in VARIANTS
            }
        )
        heatmap_dim.to_csv(f"{probe_dir}/{model_name}_dim_heatmap.csv")

        print(f"  -> Saved to {probe_dir}/")
