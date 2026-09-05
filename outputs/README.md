# Outputs

This directory contains the model-generated Chinese summaries and aggregate error
statistics used in the paper.

- `direct/`: zero-shot predictions from Aya Expanse, Gemma, and Qwen, provided as
  per-model and combined CSV/JSON files.
- `mbart_baseline/`: predictions from the supervised mBART baseline.
- `metrics/`: aggregate and fine-grained error rates derived from the reviewed
  annotations.

Prediction files contain only model identifiers, sample indices, and generated
summaries. Source dialogues and English/Chinese reference summaries are not
distributed. Predictions are provided for non-commercial research and
reproducibility; applicable model and dataset terms still apply.
