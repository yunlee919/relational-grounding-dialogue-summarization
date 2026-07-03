# Data

## XSAMSum (ClidSum)

This project uses the **XSAMSum** cross-lingual dialogue summarization dataset from the ClidSum benchmark (Wang et al., 2022b). Each example contains:

| Field | Description |
|-------|-------------|
| `dialogue` | English messenger-style dialogue |
| `summary` | English reference summary |
| `summary_zh` | Chinese reference summary (evaluation target) |

### Included files

- `raw/test.json` — full XSAMSum test split (819 examples)
- `splits/test_100_seed42.json` — fixed 100-example diagnostic subset (`seed=42`)

### Obtaining the data

If `raw/test.json` is not present, download the ClidSum/XSAMSum release and place the test split at `data/raw/test.json`. The repository script

```bash
python scripts/prepare_sample.py
```

recreates `splits/test_100_seed42.json` from the full test file.

### Training data

The mBART baseline (`jjnhuang/mbart-large-50-en-dialogue-to-zh-summary`) is a publicly released checkpoint fine-tuned on XSAMSum training data. Re-training is not required to reproduce our inference results.
