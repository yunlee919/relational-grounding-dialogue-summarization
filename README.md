# Relational Grounding Failures in Zero-Shot Cross-Lingual Dialogue Summarization with Small Language Models

**Accepted at GroundLM 2026, an EMNLP 2026 Workshop**

Code, annotations, model outputs, and reproducibility resources for our paper **“Relational Grounding Failures in Zero-Shot Cross-Lingual Dialogue Summarization with Small Language Models,”** accepted at **GroundLM 2026: Grounding Language Models: Learning Faithfully and Efficiently**, an **EMNLP 2026 Workshop**.

**Workshop:** [GroundLM 2026 — EMNLP 2026 Workshop](https://groundlm.github.io/grouplm_emnlp2026/)

**Authors:** Yunwoo Lee, Ziwa Li, Jennifer Huang
**Contact:** `{yunu919, lizhihua, jjnhuang}@uw.edu`

This repository retains the original project name `grounded-semantic-pipeline`.

## Overview

Reference-based automatic metrics such as ROUGE, BERTScore, and OmniScore may obscure source-grounded faithfulness failures where a summary preserves dialogue-relevant content but misrepresents **who did what, under what conditions, or with what outcome**.

We study zero-shot English-to-Chinese dialogue summarization with small language models (SLMs) and introduce a **Dialogue-Semantic Grounding Error Taxonomy** for fine-grained source-grounded diagnosis.

This repository provides:

* A 9-label taxonomy consisting of six hallucination categories (`H-ENT`, `H-EVT`, `H-ROLE`, `H-CIRC`, `H-MOD`, `H-DISC`), plus `OMIT`, `LANG`, and `NO-ERROR`
* A reproducible definition of a fixed 100-example evaluation subset (`seed=42`) from the **XSAMSum Dataset**
* Reproducible scripts for model inference, automatic evaluation, and error analysis
* Final taxonomy annotations and automatic metric scores for all reported systems
* Precomputed model outputs and metric tables
* Notebooks and figures for qualitative and quantitative analysis

## Key Findings

On the fixed 100-dialogue evaluation subset:

* Hallucinations produced by the SLMs are often **relational rather than arbitrary**, preserving dialogue-relevant content while altering participant roles, events, circumstances, modality, or discourse outcomes.
* **Speaker-role hallucination** is the most frequent hallucination type across the three SLMs.
* The three zero-shot SLMs produce substantially fewer omissions than the supervised mBART baseline, but each model exhibits a distinct grounding-error profile.
* mBART achieves the highest ROUGE and BERTScore values despite having the lowest `NO-ERROR` rate and the highest hallucination and omission rates under the manual taxonomy.
* These discrepancies motivate using source-grounded taxonomy analysis alongside automatic metrics rather than treating reference similarity as a direct measure of source-grounded faithfulness.

These results are descriptive of the evaluated setting and should not be interpreted as a controlled performance ranking between the supervised mBART baseline and the zero-shot SLMs.

## Repository Structure

```text
.
├── annotations/          # Final taxonomy labels + metrics (400 rows = 4 models x 100)
├── data/
│   ├── raw/              # User-provided XSAMSum data (not distributed)
│   └── splits/           # Locally generated evaluation subset
├── figures/              # Main + appendix figures
├── notebooks/            # Reproducibility and analysis notebooks
├── outputs/              # Precomputed model outputs and metric tables
├── prompts/              # Paper-aligned direct summarization prompt
├── scripts/              # CLI entry points
├── src/groundlm/         # Python package
└── taxonomy/             # Taxonomy definitions and representative examples
```

## Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
```

### 2. Inspect Published Results

The released annotations, metric tables, and figures can be inspected without re-running model inference.

```bash
python scripts/analyze_error_labels.py
python scripts/export_figures.py
```

You can also open the notebooks in `notebooks/` for result inspection, metric checking, and qualitative analysis.

### 3. Reproduce Model Outputs and Evaluation

First, prepare the fixed evaluation subset:

```bash
python scripts/prepare_sample.py
```

Pull the three SLMs used in the paper:

```bash
ollama pull qwen3.5:9b
ollama pull gemma4:e4b
ollama pull aya-expanse:8b
```

Run SLM and mBART inference:

```bash
python scripts/run_direct_slms.py --model all
python scripts/run_mbart.py
```

Run reference-based automatic evaluation:

```bash
python scripts/evaluate_rouge_bertscore.py \
  --input outputs/direct/all_models/direct_all_models_100samples_seed42_results.csv

python scripts/evaluate_omniscore.py
```

## Models

| Model          | Setting                        | Identifier                                          |
| -------------- | ------------------------------ | --------------------------------------------------- |
| Aya Expanse 8B | Zero-shot direct               | `aya-expanse:8b`                                    |
| Gemma 4 E4B    | Zero-shot direct               | `gemma4:e4b`                                        |
| Qwen 3.5 9B    | Zero-shot direct               | `qwen3.5:9b`                                        |
| mBART-large-50 | Supervised fine-tuned baseline | `jjnhuang/mbart-large-50-en-dialogue-to-zh-summary` |

The mBART baseline is fine-tuned on the XSAMSum training data, while the three SLMs are evaluated in a zero-shot direct setting.

All SLMs use the same paper-aligned prompt in `prompts/direct_prompt.txt`.

## SLM Inference Configuration

All three SLMs are evaluated locally through Ollama using the same direct zero-shot setup:

* Temperature: `0.2`
* Context window: `8192` tokens
* Maximum generated tokens: `1024`
* Explicit reasoning output: disabled
* No demonstrations
* No intermediate translation
* No task-specific fine-tuning

Given an English messenger-style dialogue, each model is instructed to generate only a concise Simplified Chinese summary.

## Dialogue-Semantic Grounding Error Taxonomy

The taxonomy is designed to identify source-grounded failures involving dialogue-semantic relations rather than relying only on reference similarity.

| Label      | Core Annotation Trigger                                                                         |
| ---------- | ----------------------------------------------------------------------------------------------- |
| `H-ENT`    | Wrong or unsupported person, entity, name, or object                                            |
| `H-EVT`    | Added, changed, or fabricated event/action                                                      |
| `H-ROLE`   | Correct event but wrong actor, recipient, speaker, or role                                      |
| `H-CIRC`   | Wrong time, place, number, quantity, condition, or schedule                                     |
| `H-MOD`    | Changed negation, uncertainty, intention, obligation, or modal force                            |
| `H-DISC`   | Misstated cause-effect, contrast, order, decision, agreement, conclusion, or outcome            |
| `OMIT`     | Missing salient event, participant, reason, decision, or outcome                                |
| `LANG`     | English leakage, malformed Chinese, repetition, translation/format issue, or non-summary output |
| `NO-ERROR` | No grounding, omission, language, format, or instruction-following error                        |

The `H-*` categories represent hallucination types. `OMIT` captures missing salient source content, while `LANG` captures target-language, formatting, and task-fulfillment failures.

Summaries may receive multiple labels when independent error types co-occur.

Full definitions are available in `taxonomy/taxonomy.csv`. Representative examples are provided in `taxonomy/table6_examples.csv`.

## Annotation Workflow

The dialogue-semantic grounding annotations were produced using a **model-assisted first-pass workflow**.

An annotation model first proposed candidate labels for each generated summary. The annotation model was not used as an evaluated summarization system or as the final annotator.

Three annotators then reviewed the proposed labels against:

* The English source dialogue
* The Chinese reference summary
* The generated Chinese summary
* The taxonomy definitions

Annotation prioritized **source-grounded faithfulness**. Hallucination labels were assigned only when a generated summary introduced or altered information relative to the source dialogue. `OMIT` and `LANG` were used for missing salient content and language/output-form failures, respectively.

Multiple labels were allowed for independent error types, and uncertain cases were discussed before finalizing the annotations.

Released annotations are available in `annotations/`, with taxonomy definitions and representative examples in `taxonomy/`.

## Data Source

This project uses the **XSAMSum Dataset** from the [ClidSum](https://github.com/krystalan/ClidSum) benchmark.

XSAMSum pairs English messenger-style dialogues with abstractive summaries and professional target-language translations. We use the **English dialogues as inputs** and the **Chinese summaries as references**.

For diagnostic evaluation, we use a fixed random sample of **100 examples from the XSAMSum test split**, selected with `seed=42`.

| Resource | Link                                                                     |
| -------- | ------------------------------------------------------------------------ |
| GitHub   | [krystalan/ClidSum](https://github.com/krystalan/ClidSum)                |
| Paper    | [ClidSum (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.526.pdf) |

The original XSAMSum data are **not redistributed in this repository**.

After obtaining the dataset from its original source, the fixed evaluation subset can be generated locally at:

```text
data/splits/test_100_seed42.json
```

See `data/README.md` for access and reproduction instructions.

Please cite the original ClidSum and XSAMSum work when reusing the dataset or derived resources.

## Automatic Evaluation

### ROUGE and BERTScore

We evaluate generated summaries against the Chinese reference summaries using:

* **ROUGE**, for lexical overlap
* **BERTScore**, for contextual embedding-based similarity

These metrics provide complementary reference-based indicators but do not directly determine whether a summary is fully supported by the source dialogue.

### OmniScore

We additionally use **OmniScore**, a reference-based learned multilingual metric for summary quality and faithfulness.

OmniScore evaluates each generated summary relative to the Chinese reference along four dimensions:

* `informativeness`
* `clarity`
* `plausibility`
* `faithfulness`

| Resource         | Link                                                                            |
| ---------------- | ------------------------------------------------------------------------------- |
| Model checkpoint | [`QCRI/OmniScore-deberta-v3`](https://huggingface.co/QCRI/OmniScore-deberta-v3) |
| Paper            | [OmniScore (arXiv:2604.05083)](https://arxiv.org/pdf/2604.05083)                |

Our evaluation script loads the checkpoint through Hugging Face:

```text
scripts/evaluate_omniscore.py
```

Released annotation tables include all four OmniScore dimensions.

## Scope and Limitations

The results in this repository should be interpreted within the scope of the study:

* A fixed 100-dialogue sample from XSAMSum
* One language direction: English-to-Chinese
* One direct zero-shot prompt
* Three instruction-tuned SLMs under 10B parameters
* One supervised mBART baseline
* A single Chinese reference summary per example

The taxonomy was induced from preliminary outputs in this setting, so its observed label distributions may vary across datasets, language pairs, model families, prompts, or decoding configurations.

Because mBART is supervised while the SLMs are evaluated zero-shot with different model and decoding configurations, cross-system differences should not be interpreted as a controlled performance ranking.

The study also does not include an English monolingual control, so translation-related and summarization-related errors cannot always be fully separated.

## Citation

If you use this repository, taxonomy, annotations, or evaluation resources, please cite our paper.

Citation information will be added after publication in the ACL Anthology.

## License

MIT License.

The XSAMSum Dataset and related ClidSum resources remain subject to their original licenses and terms of use.
