from dotenv import load_dotenv
import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from models import ModelHuggingFace
from helper import get_variants

VARIANTS = get_variants()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on harmful and harmless requests using a HuggingFace model."
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
        "--k",
        type=int,
        default=10,
        help="Number of responses to generate per request.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Recompute inference even if cached files already exist.",
    )
    return parser.parse_args()


def run_inference(
    requests,
    variant,
    model_name,
    target_model,
    request_type,
    k,
    generation_config,
    out_dir,
    load_cached=True,
):
    out_file = os.path.join(out_dir, f"{model_name}_{variant}.csv")

    if load_cached and os.path.isfile(out_file):
        print(f"Loading cached results: {out_file}")
        return

    print(f"Computing inference: {out_file}")
    rows = []
    for request in tqdm(requests):
        row_dict = {"request": request}
        responses = target_model.get_response_batched(request, generation_config)
        for j, response in enumerate(responses, start=1):
            row_dict[f"answer_{j}"] = response
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    df.to_csv(out_file, sep=";")


def main():
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    target_model = ModelHuggingFace(args.model_name, args.cache_dir, hf_token)
    print("Model loaded.")

    harmful_df = pd.read_csv("data/jbb_harmful_variants", delimiter=";")
    harmless_df = pd.read_csv("data/jbb_harmless_variants", delimiter=";")

    with open("model_configs.json", "r") as f:
        generation_config = json.load(f)

    load_cached = not args.no_cache

    for variant in VARIANTS:
        for request_type, df in [("harmful", harmful_df), ("harmless", harmless_df)]:
            out_dir = f"behavioral_eval/k={args.k}_answers_to_{request_type}_requests"
            os.makedirs(out_dir, exist_ok=True)

            run_inference(
                requests=df[variant].values,
                variant=variant,
                model_name=args.model_name,
                target_model=target_model,
                request_type=request_type,
                k=args.k,
                generation_config=generation_config,
                out_dir=out_dir,
                load_cached=load_cached,
            )
            print(f"Processed {variant}-{request_type} requests.")


if __name__ == "__main__":
    main()
