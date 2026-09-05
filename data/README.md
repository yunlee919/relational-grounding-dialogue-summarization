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
- GitHub: [krystalan/ClidSum](https://github.com/krystalan/ClidSum)
- Paper: [ClidSum (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.526.pdf)

### Expected local files

- `raw/test.json` — locally obtained XSAMSum test split (819 examples)
- `splits/test_100_seed42.json` — locally generated 100-example diagnostic subset (`seed=42`)

These data files are not distributed in this repository. See `raw/README.md`
and `splits/README.md` for details.

### Obtaining the data

Request XSAMSum access by following the instructions in the original ClidSum
repository, then place the test split at `data/raw/test.json`. The repository script

```bash
python scripts/prepare_sample.py
```

recreates `splits/test_100_seed42.json` from the full test file.

### Baseline checkpoint

The mBART baseline (`jjnhuang/mbart-large-50-en-dialogue-to-zh-summary`) is a publicly released checkpoint fine-tuned on XSAMSum training data. Re-training is not required to reproduce our inference results.
