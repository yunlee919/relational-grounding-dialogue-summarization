# Evaluation split

This directory is reserved for the paper's fixed 100-example XSAMSum evaluation
subset. The subset itself is not redistributed in this repository.

Starting from `data/raw/test.json`, the project samples 100 record indices with
Python's `random.Random(42)` and sorts the selected indices to preserve source
order. Recreate it with:

```bash
python scripts/prepare_sample.py
```

The command writes `data/splits/test_100_seed42.json`. Output files use
`sample_index` values `0`–`99` to refer to positions in this generated subset.
Do not commit or redistribute the generated split.
