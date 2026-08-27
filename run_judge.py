from dotenv import load_dotenv
import argparse
import os
import pandas as pd
from wildguard import load_wildguard
from helper import get_variants

VALID_VARIANTS = get_variants()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge model responses for harmfulness using WildGuard."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the HuggingFace model whose outputs are to be judged.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of responses per request that were generated (default: 10).",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Recompute judgements even if cached files already exist.",
    )
    return parser.parse_args()


def judge_wildguard(
    model_name, request_type, variant, k, in_dir, out_dir, load_cached=True
):
    out_file = os.path.join(out_dir, f"{model_name}_{variant}.csv")

    if load_cached and os.path.isfile(out_file):
        print(f"Loading cached results: {out_file}")
        return

    in_file = os.path.join(in_dir, f"{model_name}_{variant}.csv")
    try:
        df = pd.read_csv(in_file, sep=";")
    except FileNotFoundError:
        print(f"Skipping (not found): {in_file}")
        return

    print(f"Computing judgements: {out_file}")
    wildguard = load_wildguard()

    all_items_to_classify = []
    for _, row in df.iterrows():
        for j in range(1, k + 1):
            all_items_to_classify.append(
                {
                    "prompt": row["request"],
                    "response": row[f"answer_{j}"],
                }
            )

    all_results = wildguard.classify(all_items_to_classify)
    print(f"Classified all prompts for {model_name} - {request_type} - {variant}.")

    new_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        new_row = {"prompt": row["request"]}
        for j in range(1, k + 1):
            flat_idx = (i * k) + (j - 1)
            res = all_results[flat_idx]
            new_row[f"response_{j}"] = row[f"answer_{j}"]
            new_row[f"response_harmfulness_{j}"] = res["response_harmfulness"]
            new_row[f"response_refusal_{j}"] = res["response_refusal"]
            new_row[f"prompt_harmfulness_{j}"] = res["prompt_harmfulness"]
        new_rows.append(new_row)

    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(out_file, sep=";")
    print(f"Judged: {variant}.")


def main():
    args = parse_args()

    load_dotenv()

    load_cached = not args.no_cache

    for variant in args.variants:
        for request_type in ["harmful", "harmless"]:
            in_dir = f"behavioral_eval/k={args.k}_answers_to_{request_type}_requests"
            out_dir = f"behavioral_eval/judge_wildguard_{request_type}"
            os.makedirs(out_dir, exist_ok=True)

            judge_wildguard(
                model_name=args.model_name,
                request_type=request_type,
                variant=variant,
                k=args.k,
                in_dir=in_dir,
                out_dir=out_dir,
                load_cached=load_cached,
            )


if __name__ == "__main__":
    main()
