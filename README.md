# relational-grounding-dialogue-summarization

**Relational Grounding Failures in Zero-Shot Cross-Lingual Dialogue Summarization with Small Language Models**

Code and data for our EMNLP workshop submission on diagnosing relational grounding failures in English-to-Chinese dialogue summarization. The workshop submission title is **GroundLM**, but this repository keeps the main project name `grounded-semantic-pipeline`.

**Authors:** Yunwoo Lee, Ziwa Li, Jennifer Huang  
**Contact:** `{yunu919, jjnhuang, lizhihua}@uw.edu`

## Overview

Automatic metrics such as ROUGE, BERTScore, and OmniScore can miss faithfulness failures where a summary preserves dialogue-relevant content but misrepresents **who did what, under what conditions, or with what outcome**. This repository provides:

- A 9-label taxonomy for source-grounded diagnosis (`H-ENT` to `H-DISC`, plus `OMIT`, `LANG`, `NO-ERROR`)
- A reproducible definition of a fixed 100-example evaluation subset (`seed=42`) from the **XSAMSum Dataset**
- Reproducible scripts for generation, automatic evaluation, and error analysis
- Precomputed outputs for all reported models
- Useful notebooks for inspection, metric checking, and qualitative analysis

## Repository structure

```text
.
├── annotations/          # Final taxonomy labels + metrics (400 rows = 4 models x 100)
├── data/
│   ├── raw/              # User-provided XSAMSum data (not distributed)
│   └── splits/           # Locally generated evaluation subset
├── figures/              # Main + appendix figures
├── notebooks/            # Lightweight reproducibility and analysis notebooks
├── outputs/              # Precomputed model outputs and metric tables
├── prompts/              # Paper-aligned direct summarization prompt
├── scripts/              # CLI entry points
├── src/groundlm/         # Python package
└── taxonomy/             # Taxonomy definitions and Table 6 examples
```

## Quick start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
```

### 2. Inspect published results

```bash
python scripts/analyze_error_labels.py
python scripts/export_figures.py
```

You can also open the notebooks in `notebooks/` for result inspection and qualitative analysis without re-running the full pipeline.

### 3. Reproduce the full pipeline

```bash
python scripts/prepare_sample.py

ollama pull qwen3.5:9b
ollama pull gemma4:e4b
ollama pull aya-expanse:8b
python scripts/run_direct_slms.py --model all

python scripts/run_mbart.py

python scripts/evaluate_rouge_bertscore.py --input outputs/direct/all_models/direct_all_models_100samples_seed42_results.csv
python scripts/evaluate_omniscore.py
```

## Models

| Model | Setting | Identifier |
|-------|---------|------------|
| Aya Expanse 8B | Zero-shot direct | `aya-expanse:8b` |
| Gemma 4 E4B | Zero-shot direct | `gemma4:e4b` |
| Qwen 3.5 9B | Zero-shot direct | `qwen3.5:9b` |
| mBART-large-50 | Fine-tuned baseline | `jjnhuang/mbart-large-50-en-dialogue-to-zh-summary` |

All SLMs use the paper-aligned prompt in `prompts/direct_prompt.txt`.

## Dialogue-Semantic Grounding Error Taxonomy

| Label | Core trigger |
|-------|--------------|
| H-ENT | Wrong or unsupported person, entity, name, or object |
| H-EVT | Added, changed, or fabricated event/action |
| H-ROLE | Correct event but wrong actor, recipient, speaker, or role |
| H-CIRC | Wrong time, place, number, quantity, condition, or schedule |
| H-MOD | Changed negation, uncertainty, intention, obligation, or modal force |
| H-DISC | Misstated cause-effect, contrast, order, decision, agreement, conclusion, or outcome |
| OMIT | Missing salient event, participant, reason, decision, or outcome |
| LANG | English leakage, malformed Chinese, repetition, or non-summary output |
| NO-ERROR | No grounding, omission, language, or format error |

Full definitions: `taxonomy/taxonomy.csv`. Representative examples: `taxonomy/table6_examples.csv`.

## Data source

This project uses the **XSAMSum Dataset** from the [ClidSum](https://github.com/krystalan/ClidSum) benchmark. Please cite the original work when reusing these files or derived annotations.

| Resource | Link |
|----------|------|
| GitHub | [krystalan/ClidSum](https://github.com/krystalan/ClidSum) |
| Paper | [ClidSum (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.526.pdf) |

The fixed subset used in our paper is generated locally at `data/splits/test_100_seed42.json`; it is not distributed in this repository. See `data/README.md` for access and reproduction notes.

## OmniScore

Automatic evaluation in this repository also uses **OmniScore**, a learned multilingual metric for summary quality and faithfulness. Please cite and attribute OmniScore when reusing these evaluation results.

| Resource | Link |
|----------|------|
| Model checkpoint | [`QCRI/OmniScore-deberta-v3`](https://huggingface.co/QCRI/OmniScore-deberta-v3) |
| Paper | [OmniScore (arXiv:2604.05083)](https://arxiv.org/pdf/2604.05083) |

Our scripts load the checkpoint via Hugging Face (`scripts/evaluate_omniscore.py`). Released annotation tables include OmniScore dimensions (`informativeness`, `clarity`, `plausibility`, `faithfulness`) computed with this model.

## License

MIT License. The XSAMSum Dataset and related ClidSum resources remain subject to their original licenses and terms of use.
