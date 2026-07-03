# GroundLM

**Relational Grounding Failures in Zero-Shot Cross-Lingual Dialogue Summarization with Small Language Models**

Code and data for our EMNLP workshop paper on diagnosing *relational* grounding failures in English-to-Chinese dialogue summarization. We introduce the **Dialogue-Semantic Grounding Error Taxonomy** and apply it to three open-weight SLMs (Aya Expanse 8B, Gemma 4 E4B, Qwen 3.5 9B) alongside a supervised mBART baseline on XSAMSum.

## Overview

Automatic metrics (ROUGE, BERTScore, OmniScore) can miss faithfulness failures where a summary preserves dialogue-relevant content but misrepresents **who did what, under what conditions, or with what outcome**. GroundLM provides:

- A **9-label taxonomy** for source-grounded diagnosis (`H-ENT` … `H-DISC`, `OMIT`, `LANG`, `NO-ERROR`)
- A **fixed 100-example evaluation subset** (seed 42) from the XSAMSum test split
- **Reproducible scripts** for generation, automatic evaluation, and error analysis
- **Precomputed outputs** so results can be inspected without re-running models

## Repository structure

```
.
├── annotations/          # Final taxonomy labels + metrics (400 rows = 4 models × 100)
├── data/
│   ├── raw/              # XSAMSum test split (download instructions in data/README.md)
│   └── splits/           # Fixed 100-sample subset (seed 42)
├── figures/              # Paper figures (main + appendix)
├── outputs/              # Precomputed model outputs and metric tables
├── paper/                # ACL LaTeX source
├── prompts/              # Shared direct summarization prompt
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

### 2. Inspect published results (no GPU required)

Precomputed summaries and annotations are included:

```bash
# Aggregated error rates (Table 2 style)
python scripts/analyze_error_labels.py

# Regenerate appendix figures
python scripts/export_figures.py
```

### 3. Reproduce the full pipeline

```bash
# Create the 100-example subset (already committed; re-run to verify)
python scripts/prepare_sample.py

# SLMs via Ollama (requires local models: qwen3.5:9b, gemma4:e4b, aya-expanse:8b)
ollama pull qwen3.5:9b
ollama pull gemma4:e4b
ollama pull aya-expanse:8b
python scripts/run_direct_slms.py --model all

# mBART supervised baseline (HuggingFace)
python scripts/run_mbart.py

# Automatic metrics
python scripts/evaluate_rouge_bertscore.py --input outputs/direct/all_models/direct_all_models_100samples_seed42_results.csv
python scripts/evaluate_omniscore.py
```

## Dialogue-Semantic Grounding Error Taxonomy

| Label | Core trigger |
|-------|----------------|
| H-ENT | Wrong or unsupported person, entity, name, or object |
| H-EVT | Added, changed, or fabricated event/action |
| H-ROLE | Correct event but wrong actor, recipient, speaker, or role |
| H-CIRC | Wrong time, place, number, quantity, condition, or schedule |
| H-MOD | Changed negation, uncertainty, intention, obligation, or modal force |
| H-DISC | Misstated cause-effect, contrast, order, decision, agreement, conclusion, or outcome |
| OMIT | Missing salient event, participant, reason, decision, or outcome |
| LANG | English leakage, malformed Chinese, repetition, or non-summary output |
| NO-ERROR | No grounding, omission, language, or format error |

Full definitions: `taxonomy/taxonomy.csv`. Representative source–output pairs: `taxonomy/table6_examples.csv`.

## Models

| Model | Setting | Ollama / HF identifier |
|-------|---------|------------------------|
| Aya Expanse 8B | Zero-shot direct | `aya-expanse:8b` |
| Gemma 4 E4B | Zero-shot direct | `gemma4:e4b` |
| Qwen 3.5 9B | Zero-shot direct | `qwen3.5:9b` |
| mBART-large-50 | Supervised fine-tuned | `jjnhuang/mbart-large-50-en-dialogue-to-zh-summary` |

All SLMs share the prompt in `prompts/direct_prompt.txt` (temperature 0.2, `num_ctx` 8192).

## Data

We evaluate on **XSAMSum** from the [ClidSum](https://github.com/Yiran1010/ClidSum) benchmark (English dialogues, professional Chinese summaries). The fixed subset is `data/splits/test_100_seed42.json`. See `data/README.md` for obtaining the full test split.

## Citation

```bibtex
@inproceedings{groundlm2026,
  title={Relational Grounding Failures in Zero-Shot Cross-Lingual Dialogue Summarization with Small Language Models},
  author={Anonymous},
  booktitle={EMNLP Workshop},
  year={2026}
}
```

Update the author field after deanonymization.

## License

MIT License. XSAMSum/ClidSum data are subject to their original licenses.
