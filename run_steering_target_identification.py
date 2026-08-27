import argparse
import os
import pickle
from collections import Counter

import pandas as pd
from helper import get_variants, get_valid_models

VALID_MODELS = get_valid_models()
VALID_VARIANTS = get_variants()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive per-prompt steering targets (complied/refusal) from WildGuard judge results."
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=VALID_MODELS,
        help="Name of the model whose judge results are to be processed.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of responses per request that were judged (default: 10).",
    )
    return parser.parse_args()


def majority(row, cols):
    return Counter([row[c] for c in cols]).most_common(1)[0][0]


def get_steering_targets(model_name, request_type, variant, k):
    in_file = os.path.join(
        f"results/judge_wildguard_{request_type}", f"{model_name}_{variant}.csv"
    )
    try:
        df = pd.read_csv(in_file, sep=";")
    except FileNotFoundError:
        print(f"Skipping (not found): {in_file}")
        return None

    refusal_cols = [f"response_refusal_{j}" for j in range(1, k + 1)]
    hf_cols = [f"response_harmfulness_{j}" for j in range(1, k + 1)]

    # Even in greedy mode, outputs per inference vary slightly, therefore prompts are deemed worth steering,
    # if the majority of their 10 generations was refusal/compliance.

    labels = {}
    for i, row in df.iterrows():
        r = majority(row, refusal_cols)
        h = majority(row, hf_cols)
        labels[i] = (h, r)

    if request_type == "harmful":
        targets = [
            "complied" if label_tuple == ("harmful", "compliance") else "refusal"
            for label_tuple in labels.values()
        ]
    else:
        targets = [
            "complied" if label_tuple[1] == "compliance" else "refusal"
            for label_tuple in labels.values()
        ]

    n_complied = sum(1 for t in targets if t == "complied")
    print(f"{request_type}: {variant} -> {n_complied} complied.")
    return targets


def main():
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

    result = {"harmful": {}, "harmless": {}}

    for variant in VALID_VARIANTS:
        for request_type in ["harmful", "harmless"]:
            targets = get_steering_targets(
                model_name=model_name,
                request_type=request_type,
                variant=variant,
                k=args.k,
            )
            if targets is not None:
                result[request_type][variant] = targets

    out_dir = f"steering/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"steering_targets.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved steering targets: {out_file}")


if __name__ == "__main__":
    main()
