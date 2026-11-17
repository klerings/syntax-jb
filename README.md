# Mood Matters: How Syntactic Sensitivity Undermines Safety Alignment

[[Paper]]([https://arxiv.org/abs/2608.05409](https://aclanthology.org/2026.emnlp-main.317/))

> *Syntactic Vulnerability*: inability to reject harmful requests expressed differently than the default imperative form  
  
<img src="example_prompt.png" alt="syntactic_vulnerability" width="600"/>

_Above: By changing something as trivial as the grammatical mood from **imperative** to **interrogative** (asking a question instead of giving a command), the refusal mechanism is disabled. Through steering a syntactic "imperative" feature, it can be re-instated because models have learned spurious correlations rather than pure semantic triggers for harmfulness._

## Overview

- We show that syntactic vulnerability is a general failure mode affecting 16 aligned LLMs from 7B to 70B parameters.
- We demonstrate through causal mediation analysis and steering that refusal behavior partially depends on upstream syntactic features independent of harmful intent.
- We trace this vulnerability to syntactic biases in post-training data and show that increasing syntactic diversity substantially improves robustness without degrading capabilities.  
  

> **This repository is not independently runnable.** The syntactically paraphrased prompts of the JBB dataset are not released (see [Responsible Disclosure](#responsible-disclosure)). The code here documents and reproduces the pipeline structure, but §2–§5 cannot be executed end-to-end without that data. Contact the authors to request access.

## Contents

- [Setup](#setup)
- [Common Arguments](#common-arguments)
- [§2 Behavioral Evaluation](#-2-behavioral-evaluation)
    - [Answer Generation](#answer-generation-on-harmfulharmless-prompt-variants-of-jbb-behaviors)
    - [LLM-as-a-Judge](#llm-as-a-judge-wildguard)
- [§3 Harmfulness Probing](#-3-harmfulness-probing)
    - [Extraction of Residual Activations](#extraction-of-residual-activations-layer-wise-on-harmfulharmless-jbb-prompt-variants)
    - [Diff-in-Means and Cluster Probing](#difference-in-means-and-cluster-probing)
- [§4 Causal Mediation Analysis & Steering](#-4-causal-mediation-analysis--steering)
    - [Feature Identification](#feature-identification)
    - [Steering](#steering)
    - [Summary of Steering](#summary-of-steering)
- [§5 Post-Training Dataset Analysis](#-5-post-training-dataset-analysis)
    - [Mood Distribution](#mood-distribution)
    - [Voice Distribution](#voice-distribution)
- [§6 Post-Training on Balanced Corpora](#-6-post-training-on-balanced-corpora)
- [Responsible Disclosure](#responsible-disclosure)
- [Citation](#citation)
- [License](#license)

## Setup
Environment: Python 3.13 via `uv sync`.  
Set the following in `.env`:
- `HF_TOKEN`
- `NEURONPEDIA_API_KEY`

## Common Arguments
 
The following arguments recur across multiple scripts; script-specific tables below list only additional flags.
 
| Argument | Values | Used in |
|---|---|---|
| `model_name` | Model identifier from `model_configs.json` | §2, §3, §4 |
| `cache_dir` | Path to model weight cache | §2, §3, §4 |
| `--k` | Number of responses per request (default: 10) | §2, §4 |
| `--no_cache` | Recompute even if output files exist | §2, §3, §4 |
 
---

## § 2: Behavioral Evaluation

### Answer Generation on harmful/harmless prompt variants of JBB-Behaviors

```bash
python run_jbb.py <model_name> <cache_dir> [--k <int>] [--no_cache]
```

Results: `behavioral_eval/k=<k>_answers_to_<request_type>_requests/<model_name>_<variant>.csv`


### LLM-as-a-Judge (WildGuard)

```bash
python run_judge.py <model_name> [--k <int>] [--no_cache]
```

Results: `behavioral_eval/judge_wildguard_<request_type>/<model_name>_<variant>.csv`


## § 3: Harmfulness Probing

### Extraction of Residual Activations Layer-Wise on harmful/harmless JBB prompt variants
 
```bash
python run_residual_collection.py <model_name> <cache_dir> [--position <pos>] [--no_cache]
```

Results: `probing/resids_<position>/<model_name>/<request_type>_<variant>.pt`
 
| Argument | Values |
|---|---|
| `--position` | `mean` (default), `final`, `last_prompt`, `first`, `bos`|
 
> **Note:** the position used here must match `--position` in the probing step below because the probe reads from `probing/resids_<position>/`.

### Difference-in-Means and Cluster Probing

Residual activations are probed for harmfulness using a diff-in-means probe and a cluster probe, evaluated across all (train variant × test variant) syntax combinations.

**Fit probes:**
```bash
python run_probes.py --position mean --pred hard --cv
```

Results: `probing/variant_probes_{position}_{pred}_{cv5|split}/`
 
**Plot results:**
```bash
python run_probing_plots.py --position mean --pred hard --cv
```

Produces a cluster accuracy heatmap and a layerwise accuracy plot (with ±1 std shading when using CV) per model, saved to `probing/plots_{position}_{pred}_{cv5|split}/`.
 
**Key options:**
 
| Argument | Description |  
|---|---|  
| `--position` | Token position: `mean` (default), `final`, `last_prompt`, `first`, `bos` |  
| `--pred` | Diff-in-means decision rule: `hard` (logit > 0, default) or `soft` (sigmoid > 0.5) |  
| `--cv` | Use 5-fold cross-validation instead of a single 70/30 split |   
| `--n_folds` | Number of CV folds (default=5) |  
| `--models` | Space-separated list of models to run |  
 
The results reported in the paper use `--position mean --pred hard --cv` (5-fold CV).

## § 4: Causal Mediation Analysis & Steering

Currently, this analysis is only enabled for Gemma 2 (2B) & Qwen 2.5 (7B)  & Llama 3.1 (8B). It requires prior **greedy** generation and judgement of answers with `gemma2-2b-greedy`/`qwen2.5_7b_greedy`/`llama3.1-8b-greedy`.

### Feature Identification

**Identify Steerable Prompts** - Derive per-prompt steering targets (complied/refusal) from WildGuard judge results.

```bash
python run_steering_target_identification.py <model_name> [--k <int>] 
```

Results: `steering/<model_name>/steering_targets.pkl`

> **Note:** currently allowed models: `gemma2-2b-greedy`,`qwen2.5_7b_greedy`,`llama3.1-8b-greedy`

**Determine Top50** - Identify top-50 upstream SAE features that influence refusal behaviour and react to syntax, using gradient-based scoring and grammatical mood activation differences.

```bash
python run_upstream_feature_identification.py <model_name> <cache_dir> --upstream_layer <int> [--no_cache]
```

Results: `sae_cache/<model>/top50_{promotion,suppression}_features_<layer>_grad_of_<label>_filtered_by_<diff_type>_mean_over_pos.csv`

| Argument | Values |
|---|---|
| `--upstream_layer` | Layer index to search for refusal-influencing features (required) |

**Select Steering Coefficients** - For each upstream layer 1-5, manually select syntax/grammar-related SAE features from the top50 list by reviewing Neuronpedia interpretations in `sae_cache/<model>/top50_promotion_features_<layer>_grad_of_conditional_filtered_by_mean_diff_imp-cond_mean_over_pos.csv`. Store selections in `steering/<model_name>/steering_features.json` as follows, where the key marks the layer index and the list contains the steering features:

```json
{   
  "1": [13253,11649,2328,6852,7648,9401],   
  "2": [14418,10149,6912,10360,8490,8221],   
  ...   
}   
```

Then determine the ideal steering coefficient for each feature (filter out, which features cannot meaningfully affect the refusal metric): 

```bash
python run_steering_effect_computation.py <model_name> <cache_dir> --upstream_layer <int>
```

Results: `steering/<model_name>/l<upstream_layer>_steering_config.pkl`

### Steering
Perform the final steering experiment during generation and evaluate outputs using WildGuard:

```bash
python run_steering.py <model_name> <cache_dir> --upstream_layer <int> [--random]
```

| Argument | Values |
|---|---|
| `--random` | Steer using three randomly sampled SAE latents instead of the selected features |


Results: `steering/<model_name>/generations/`, `steering/<model_name>/wildguard_eval/`.

### Summary of Steering

Print a summary of steering results across all configurations:
   

```bash
python run_steering_summary.py <model_name>
```

## § 5: Post-Training Dataset Analysis

### Mood Distribution

Compute distribution of grammatical mood for supported training datasets.
Results: printed to stdout and saved to `dataset_stats_mood/<dataset>_<phase>[_safety].txt`
 
```bash
python run_ds_analysis_mood.py <dataset> <phase> [--safety]
```

### Voice Distribution

Compute distribution of grammatical voice for supported training datasets.
Results: printed to stdout and saved to `dataset_stats_voice/<dataset>_<phase>[_safety].json`
 
```bash
python run_ds_analysis_voice.py <dataset> <phase> [--safety]
```
 
| Argument | Values |
|---|---|
| `dataset` | `tulu3`, `dolci`, `apertus` |
| `phase` | `sft`, `dpo`, `rl`, `alignment` |
| `--safety` | Restrict to safety sources; adds per-source breakdown |
 
Valid dataset/phase combinations and safety support:
 
| | `sft` | `dpo` | `rl` | `alignment` |
|---|---|---|---|---|
| `tulu3` | ✓ safety | ✓ | ✓ | — |
| `dolci` | ✓ safety | ✓ safety | ✓ | — |
| `apertus` | ✓ | — | — | ✓ safety |

## § 6: Post-Training on Balanced Corpora

We release two model checkpoints on Huggingface, based on `Llama-3.1-8B` that have been trained with LoRA (rank 16, alpha 32) for 1 epoch using SFT on a 10% stratified sample of the `allenai/tulu-3-sft-mixture`:

- [ljb-llama3.1-8b-imperative](https://huggingface.co/AlinaKl/ljb-llama3.1-8b-imperative)
- [ljb-llama3.1-8b-syntax-mix](https://huggingface.co/AlinaKl/ljb-llama3.1-8b-syntax-mix)

To prevent the proliferation of harmful content, we do not release the third of our trained adapters, namely the baseline trained without any safety samples (see [Responsible Disclosure](#responsible-disclosure)).

## Responsible Disclosure

While our findings are intended to support auditing and the development of more robust alignment techniques, detailed descriptions of the failure mode could also facilitate misuse if reproduced directly. To reduce this risk, we do not release

- **Synthetic paraphrasing pipeline** (§2 data creation): the pipeline for synthetic paraphrasing of malicious prompts, and the transformed prompts themselves, are not released. The paper describes the procedure only at a high level. Contact the authors directly for access.
- **Unsafe baseline adapter** (§6): the LoRA adapter trained without any safety samples is not released.
 

## Citation 

```bibtex
@inproceedings{klerings-etal-2026-mood,
    title = "Mood Matters: How Syntactic Sensitivity Undermines Safety Alignment",
    author = "Klerings, Alina  and
      Brinkmann, Jannik  and
      Stuckenschmidt, Heiner  and
      Ponzetto, Simone Paolo",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.317/",
    doi = "10.18653/2026.emnlp-main.317",
    pages = "7316--7336",
    ISBN = "979-8-89176-459-0",
}
```


## License
Our code is licensed under the MIT license. 
