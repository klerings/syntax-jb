import argparse
import os


from dataset_analysis import (
    DATASETS,
    annotate,
    get_config,
    print_stats,
    tag_english,
)


# ---------------------------------------------------------------------------
# Unified run function
# ---------------------------------------------------------------------------


def run(dataset, phase, safety=False):
    """Load, filter, annotate and report stats for any registered dataset/phase.

    dataset : registered dataset name (e.g. 'tulu3', 'dolci', 'apertus')
    phase   : training phase (e.g. 'sft', 'dpo', 'rl', 'alignment')
    safety  : if True, restrict to safety-relevant sources and show per-source stats.
              If the phase has no safety split defined, this flag is ignored.
    """
    config, safety_sources = get_config(dataset, phase, safety=safety)

    ds = config["load"](safety_sources=safety_sources)
    ds, n_total, n_english, n_non_english = tag_english(ds)

    ds_english = ds.filter(lambda x: x, input_columns="is_english", num_proc=8)
    ds_english = annotate(ds_english)

    df = ds_english.to_pandas()
    print(
        f"\n=== {dataset.upper()} / {phase} {'(safety subset)' if safety else ''} ==="
    )

    has_source_col = "source" in df.columns
    sources = list(df["source"].unique()) if (safety and has_source_col) else None

    suffix = f"_{phase}_safety" if safety else f"_{phase}"
    os.makedirs("dataset_stats_mood", exist_ok=True)
    out_file = os.path.join("dataset_stats_mood", f"{dataset}{suffix}.txt")

    print_stats(
        df, n_total, n_english, n_non_english, sources=sources, out_file=out_file
    )
    print(f"Results written to {out_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt dataset analysis.")
    parser.add_argument("dataset", choices=sorted({d for d, _ in DATASETS}))
    parser.add_argument("phase", choices=sorted({p for _, p in DATASETS}))
    parser.add_argument(
        "--safety",
        action="store_true",
        help=(
            "Restrict to safety-relevant sources and show per-source breakdown "
            "(raises an error for phases without a safety split)"
        ),
    )
    args = parser.parse_args()

    run(args.dataset, args.phase, safety=args.safety)
