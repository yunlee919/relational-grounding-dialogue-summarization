# Data

## XSAMSum Dataset

This project uses the **XSAMSum Dataset** from the ClidSum benchmark (Wang et al., 2022b). Please cite and attribute the dataset source whenever reusing these files or derived annotations. Each example contains:

| Field | Description |
|-------|-------------|
| `dialogue` | English messenger-style dialogue |
| `summary` | English reference summary |
| `summary_zh` | Chinese reference summary (evaluation target) |

### Source attribution

- Dataset name: `XSAMSum Dataset`
- Benchmark/project: `ClidSum`
- Original benchmark page: [ClidSum](https://github.com/Yiran1010/ClidSum)

### Included files

- `raw/test.json` — full XSAMSum test split (819 examples)
- `splits/test_100_seed42.json` — fixed 100-example diagnostic subset (`seed=42`)

### Obtaining the data

If `raw/test.json` is not present, download the XSAMSum Dataset from the ClidSum release and place the test split at `data/raw/test.json`. The repository script

```bash
python scripts/prepare_sample.py
```

recreates `splits/test_100_seed42.json` from the full test file.

### Baseline checkpoint

The mBART baseline (`jjnhuang/mbart-large-50-en-dialogue-to-zh-summary`) is a publicly released checkpoint fine-tuned on XSAMSum training data. Re-training is not required to reproduce our inference results.
