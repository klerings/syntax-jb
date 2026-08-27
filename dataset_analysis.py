"""Shared NLP pipeline utilities for prompt dataset analysis.

This module provides:
  - dataset-specific loaders
  - English filtering
  - Prompt-level feature extractors (length, mood, question detection)
  - The shared annotation pipeline
  - Stats reporting
"""

import spacy
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from datasets import concatenate_datasets, load_dataset, load_from_disk


DetectorFactory.seed = 0

nlp_spacy = spacy.load("en_core_web_lg")

# ---------------------------------------------------------------------------
# Shared prompt extraction helpers
# ---------------------------------------------------------------------------


def _first_user_message(messages):
    """Return the first user-role message content from a messages list."""
    for m in messages:
        if m["role"] == "user":
            return {"prompt": m["content"]}
    print("Warning: no user message found")
    return {"prompt": ""}


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# Each loader returns a HuggingFace Dataset with at least a 'prompt' column
# and, where meaningful, a 'source' column for per-source breakdowns.
# ---------------------------------------------------------------------------


def load_tulu3_sft(safety_sources=None):
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", num_proc=8)
    if safety_sources:
        ds = ds.filter(
            lambda src: src in safety_sources, input_columns="source", num_proc=8
        )
    return ds.map(
        _first_user_message, input_columns="messages", num_proc=8
    ).select_columns(["prompt", "source"])


def load_tulu3_dpo(safety_sources=None):
    ds = load_dataset(
        "allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train", num_proc=8
    )
    return ds.select_columns(["prompt", "source"])


def load_tulu3_rl(safety_sources=None):
    ds = load_dataset(
        "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", split="train", num_proc=8
    )
    return (
        ds.map(_first_user_message, input_columns="messages", num_proc=8)
        .rename_column("dataset", "source")
        .select_columns(["prompt", "source"])
    )


def load_dolci_sft(safety_sources=None):
    def _normalise_source(raw):
        name = raw.lower()
        if "wildjailbreak" in name:
            return {"source": "wildjailbreak"}
        if "wildguardmix" in name:
            return {"source": "wildguardmix"}
        if "coconot" in name:
            return {"source": "coconot"}
        return {"source": "other"}

    ds = load_dataset("allenai/Dolci-Instruct-SFT", num_proc=8)["train"].map(
        _normalise_source,
        input_columns="source_dataset",
        num_proc=8,
        desc="Normalising source",
    )
    if safety_sources:
        ds = ds.filter(
            lambda src: src in safety_sources, input_columns="source", num_proc=8
        )
    return ds.map(
        _first_user_message, input_columns="messages", num_proc=8
    ).select_columns(["prompt", "source"])


def load_dolci_dpo(safety_sources=None):
    def _extract(chosen, prompt_id):
        if prompt_id.startswith("multiturn"):
            source = "-".join(prompt_id.split("_")[:-1])
        else:
            source = "-".join(prompt_id.split("-")[:-3])
        return {"prompt": chosen[0]["content"], "source": source}

    ds = load_dataset("allenai/Dolci-Instruct-DPO", num_proc=3)["train"]
    if safety_sources:
        ds = ds.filter(
            lambda pid: any(pid.startswith(s) for s in safety_sources),
            input_columns="prompt_id",
            num_proc=8,
        )
    return ds.map(
        _extract, input_columns=["chosen", "prompt_id"], num_proc=8
    ).select_columns(["prompt", "source"])


def load_dolci_rl(safety_sources=None):
    return load_dataset("allenai/Dolci-Instruct-RL", num_proc=4)[
        "train"
    ].select_columns(["prompt"])


def load_apertus_sft(safety_sources=None):
    cache = "data/apertus_posttraining_sft"
    if os.path.exists(f"{cache}/dataset_info.json"):
        return load_from_disk(cache)

    def _extract(messages):
        for m in messages:
            if m["role"] == "user":
                return {"prompt": m["content"]["parts"][0]["text"]}
        return {"prompt": ""}

    ds = load_dataset("swiss-ai/apertus-sft-mixture", num_proc=4)["train"].map(
        _extract, input_columns="messages", num_proc=8, desc="Extracting prompts"
    )
    ds = ds.add_column("source", ["apertus_sft"] * len(ds))
    ds = ds.select_columns(["prompt", "source"])
    ds.save_to_disk(cache)
    return ds


def load_apertus_alignment(safety_sources=None):
    ptp = load_dataset("ToxicityPrompts/PolygloToxicityPrompts", "wildchat-en")[
        "wildchat"
    ].select_columns(["prompt"])

    if safety_sources:
        return ptp

    def _extract_olmo(chosen):
        assert chosen[0]["role"] == "user"
        return {"prompt": chosen[0]["content"]}

    def _extract_prism(conversation):
        for turn in conversation:
            if turn["turn"] == 0 and turn["role"] == "user":
                return {"prompt": turn["content"]}
        return {"prompt": ""}

    olmo = (
        load_dataset(
            "allenai/olmo-2-0325-32b-preference-mix", split="train", num_proc=3
        )
        .map(_extract_olmo, input_columns="chosen", num_proc=8)
        .select_columns(["prompt"])
    )

    prism = (
        load_dataset(
            "HannahRoseKirk/prism-alignment", "conversations", split="train", num_proc=3
        )
        .filter(
            lambda t: t in ("values guided", "controversy guided"),
            input_columns="conversation_type",
            num_proc=8,
        )
        .map(_extract_prism, input_columns="conversation_history", num_proc=8)
        .select_columns(["prompt"])
    )

    return concatenate_datasets([ptp, olmo, prism])


# ---------------------------------------------------------------------------
# Dataset registry
#
# Each entry maps a (dataset, phase) pair to:
#   load           : loader function with signature load(safety_sources=None)
#   safety_sources : set of source labels that count as "safety" subset,
#                    or None if the phase has no safety split
# ---------------------------------------------------------------------------

DATASETS = {
    ("tulu3", "sft"): {
        "load": load_tulu3_sft,
        "safety_sources": {
            "ai2-adapt-dev/coconot_converted",
            "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
            "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
        },
    },
    ("tulu3", "dpo"): {
        "load": load_tulu3_dpo,
        "safety_sources": None,
    },
    ("tulu3", "rl"): {
        "load": load_tulu3_rl,
        "safety_sources": None,
    },
    ("dolci", "sft"): {
        "load": load_dolci_sft,
        "safety_sources": {"wildjailbreak", "wildguardmix", "coconot"},
    },
    ("dolci", "dpo"): {
        "load": load_dolci_dpo,
        "safety_sources": {
            "tulu_v3.9_wildjailbreak_decontaminated_50k",
            "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
            "tulu-3-sft-coconot-regenerated",
        },
    },
    ("dolci", "rl"): {
        "load": load_dolci_rl,
        "safety_sources": None,
    },
    ("apertus", "sft"): {
        "load": load_apertus_sft,
        "safety_sources": None,
    },
    ("apertus", "alignment"): {
        "load": load_apertus_alignment,
        "safety_sources": "safety_only",  # special: safety_only means use PTP subset only
    },
}


def get_config(dataset, phase, safety=False):
    """Resolve a registered (dataset, phase) config and validate the safety flag.

    Returns (config, safety_sources) where safety_sources is None unless
    safety=True and the phase has a safety split.

    Raises ValueError for unknown dataset/phase combinations and when
    safety=True is requested for a phase without a safety split.
    """
    key = (dataset, phase)
    if key not in DATASETS:
        available = ", ".join(f"{d}/{p}" for d, p in DATASETS)
        raise ValueError(
            f"Unknown dataset/phase '{dataset}/{phase}'. Available: {available}"
        )
    config = DATASETS[key]
    if safety and config["safety_sources"] is None:
        raise ValueError(f"--safety is not supported for {dataset}/{phase}")
    return config, config["safety_sources"] if safety else None


# ---------------------------------------------------------------------------
# Prompt-level feature extractors
# ---------------------------------------------------------------------------


def get_length(text):
    """Return the word count of a prompt."""
    return {"length": len(text.split())}


def get_mood(text):
    """Classify the mood of the last sentence of a prompt.

    A prompt ending with '?' or starting with a WH-word is interrogative;
    everything else is treated as imperative.
    Empty or untokenizable prompts return both flags False.
    Returns a dict with boolean flags 'imperative_adv' and 'interrogative_adv'.
    """
    doc = nlp_spacy(text)
    last_sent = doc
    if doc.has_annotation("SENT_START"):
        for sent in doc.sents:
            last_sent = sent

    if len(last_sent) == 0:
        return {"imperative_adv": False, "interrogative_adv": False}

    if last_sent.text.endswith("?"):
        return {"imperative_adv": False, "interrogative_adv": True}

    wh_tags = {"WDT", "WP", "WP$", "WRB"}
    wh_words = {"where", "who", "when", "what", "why", "whose", "which", "how"}
    first = last_sent[0]
    if first.tag_ in wh_tags or first.text.lower() in wh_words:
        return {"imperative_adv": False, "interrogative_adv": True}

    return {"imperative_adv": True, "interrogative_adv": False}


# ---------------------------------------------------------------------------
# English filtering
# ---------------------------------------------------------------------------


def _tag_english(prompt):
    """Return {'is_english': bool} for a single prompt."""
    if not prompt or not isinstance(prompt, str):
        return {"is_english": False}
    try:
        return {"is_english": detect(prompt) == "en"}
    except LangDetectException:
        return {"is_english": False}
    except Exception as e:
        print(f"Error detecting language for prompt: {prompt[:50]}... -> {e}")
        return {"is_english": False}


def tag_english(dataset):
    """Add an 'is_english' boolean column to every row without dropping anything.

    Returns (tagged_dataset, n_total, n_english, n_non_english).
    """
    dataset = dataset.map(
        _tag_english, input_columns="prompt", num_proc=12, desc="Detecting language"
    )
    n_total = len(dataset)
    n_english = sum(dataset["is_english"])
    n_non_english = n_total - n_english
    print(
        f"Language detection: {n_english} English, {n_non_english} non-English out of {n_total}"
    )
    return dataset, n_total, n_english, n_non_english


# ---------------------------------------------------------------------------
# Shared annotation pipeline
# ---------------------------------------------------------------------------


def annotate(dataset):
    """Apply all feature extractors to a dataset that has a 'prompt' column.

    Always adds: length, imperative_adv, interrogative_adv.
    Optionally adds: ends_with_q, contains_q (set include_question=True).
    """
    dataset = dataset.map(
        get_length, input_columns="prompt", num_proc=8, desc="Computing length"
    )
    dataset = dataset.map(
        get_mood, input_columns="prompt", num_proc=16, desc="Detecting mood"
    )
    return dataset


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------


def print_stats(
    dataset_or_df, n_total, n_english, n_non_english, sources=None, out_file=None
):
    """Print language and mood statistics, and optionally write to a file.

    Outputs two sections:
      1. Language counts (English vs non-English)
      2. Mood breakdown for English-only prompts, optionally per source

    dataset_or_df  : a HuggingFace Dataset or pandas DataFrame (English rows only,
                     with imperative_adv and interrogative_adv columns).
    n_total        : total prompts before language filtering.
    n_english      : number of English prompts.
    n_non_english  : number of non-English prompts.
    sources        : list of source values for per-source breakdown, or None for totals only.
    out_file       : optional path; output is written there in addition to stdout.
                     Parent directories are created automatically.
    """
    import os

    df = (
        dataset_or_df
        if hasattr(dataset_or_df, "groupby")
        else dataset_or_df.to_pandas()
    )

    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with (open(out_file, "w") if out_file else _null_context()) as f:
        _emit("=== Language ===", f)
        _emit(f"  Total        : {n_total}", f)
        _emit(f"  English      : {n_english / n_total * 100:.1f}% ({n_english})", f)
        _emit(
            f"  Non-English  : {n_non_english / n_total * 100:.1f}% ({n_non_english})\n",
            f,
        )

        _emit("=== Mood (English only) ===", f)
        rows = (list(sources) + [None]) if sources else [None]
        for source in rows:
            _print_mood_stats(df, source, f, total=n_english)


def _print_mood_stats(df, source, f, total):
    """Print mood stats for one source slice (or all rows if source is None)."""
    subset = df[df["source"] == source] if source else df
    label = source or "TOTAL"
    size = len(subset)
    if size == 0:
        _emit(f"{label}\n  (empty)\n", f)
        return
    n_imp = subset["imperative_adv"].sum()
    n_int = subset["interrogative_adv"].sum()
    denom = total
    _emit(f"{label}\n  size         : {size}", f)
    _emit(f"  Imperative   : {n_imp  / denom * 100:.1f}% ({n_imp})", f)
    _emit(f"  Interrogative: {n_int  / denom * 100:.1f}% ({n_int})\n", f)


def _emit(line, f=None):
    """Print a line to stdout and optionally to an open file handle."""
    print(line)
    if f:
        f.write(line + "\n")


class _null_context:
    """Minimal context manager that does nothing, used when no output file is requested."""

    def __enter__(self):
        return None

    def __exit__(self, *_):
        pass
