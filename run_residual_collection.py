from dotenv import load_dotenv
import argparse
import os
import torch
import pandas as pd
from models import ModelHuggingFace
from helper import get_variants

VALID_POSITIONS = ["mean", "last_prompt", "final", "first", "bos"]
VALID_VARIANTS = get_variants()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect residual activations from a HuggingFace model."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the HuggingFace model to load.",
    )
    parser.add_argument(
        "cache_dir",
        type=str,
        help="Directory to use as the HuggingFace model cache.",
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=VALID_POSITIONS,
        default="mean",
        help=f"Token position at which to extract residuals. Choices: {VALID_POSITIONS}",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Recompute residuals even if cached files already exist.",
    )
    return parser.parse_args()


def collect_residuals(model, prompts, out_file, position, load_collected=True):
    if load_collected and os.path.exists(out_file):
        print(f"Loading cached residuals: {out_file}")
        return torch.load(out_file, map_location="cpu", weights_only=False)

    print(f"Computing residuals: {out_file}")
    resid_layerwise = model.get_resids(prompts, position)
    torch.save(resid_layerwise, out_file)
    return resid_layerwise


def main():
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    model = ModelHuggingFace(args.model_name, args.cache_dir, hf_token)
    print("Model loaded.")

    harmful_df = pd.read_csv("data/jbb_harmful_variants.csv", delimiter=";")
    harmless_df = pd.read_csv("data/jbb_harmless_variants.csv", delimiter=";")

    out_dir = f"probing/resids_{args.position}/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    load_collected = not args.no_cache

    for variant in VALID_VARIANTS:
        for request_type, df in [("harmful", harmful_df), ("harmless", harmless_df)]:
            prompts = list(df[variant].values)
            out_file = os.path.join(out_dir, f"{request_type}_{variant}.pt")
            collect_residuals(model, prompts, out_file, args.position, load_collected)


if __name__ == "__main__":
    main()
