import argparse
import os
import spacy
import json

nlp_spacy = spacy.load("en_core_web_lg")


from dataset_analysis import DATASETS, get_config, tag_english


def get_sents_batch(batch_prompts):
    last_sents, sent_counts = [], []
    for doc in nlp_spacy.pipe(batch_prompts, batch_size=1000):
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
        last_sents.append(sents[-1].text)
        sent_counts.append(len(sents))
    return {"num_sents": sent_counts, "last_sent": last_sents}


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

    prompts = ds_english.map(
        function=get_sents_batch,
        input_columns="prompt",
        batched=True,
        batch_size=500,
        num_proc=8,
        desc="get_sents",
    )
    prompts = prompts.select_columns(["num_sents", "last_sent"])

    os.makedirs("dataset_stats_voice", exist_ok=True)
    if safety:
        download_dir = f"dataset_stats_voice/downloaded_datasets/{dataset}_{phase}_safety_filtered_en"
    else:
        download_dir = (
            f"dataset_stats_voice/downloaded_datasets/{dataset}_{phase}_filtered_en"
        )

    os.makedirs(download_dir, exist_ok=True)
    prompts.save_to_disk(download_dir)

    print("Extracted last sentence and filtered out non-English texts.")
    size = len(prompts)
    if size == 0:
        subset = " (safety subset)" if safety else ""
        raise ValueError(
            f"No English prompts found for {dataset}/{phase}{subset}; "
            "cannot compute voice stats."
        )

    nlp_tr = spacy.load(
        "en_core_web_trf",
        disable=["tagger", "attribute_ruler", "lemmatizer", "ner"],
    )
    num_active = 0
    num_passive = 0

    texts = prompts["last_sent"]
    BATCH_SIZE = 512
    for i, doc in enumerate(nlp_tr.pipe(texts, batch_size=BATCH_SIZE)):
        is_passive = False
        for token in doc:
            if token.dep_ == "ROOT":
                deps = {child.dep_ for child in token.children}
                is_passive = bool(deps & {"nsubjpass", "auxpass", "csubjpass"})
                break
        if is_passive:
            num_passive += 1
        else:
            num_active += 1

        if i % 50000 == 0 and i > 0:
            print(f"{i}/{size} processed", flush=True)

    print(f"Total Length: {size}")
    print(f"- # Active: {round((num_active/size)*100,1)}% ({num_active})")
    print(f"- # Passive: {round((num_passive/size)*100,1)}% ({num_passive})")

    suffix = f"_{phase}_safety" if safety else f"_{phase}"
    out_file = os.path.join("dataset_stats_voice", f"{dataset}{suffix}.json")

    with open(out_file, "w") as f:
        json.dump(
            {
                "total": size,
                "num_active": num_active,
                "num_passive": num_passive,
            },
            f,
            indent=2,
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
