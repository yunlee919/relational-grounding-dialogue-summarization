#!/usr/bin/env python3
"""Compute ROUGE and BERTScore for model outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from groundlm.metrics.rouge_bertscore import evaluate_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate summaries with ROUGE and BERTScore.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/direct/all_models/direct_all_models_100samples_seed42_results.csv"),
        help="CSV with model_name, generated_summary_zh, reference_summary_zh.",
    )
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=Path("outputs/metrics/corpus_scores_zh_XSAMSum.csv"),
    )
    parser.add_argument(
        "--pair-output",
        type=Path,
        default=Path("outputs/metrics/pair_scores_zh_XSAMSum.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    corpus_df, pair_df = evaluate_dataframe(df)

    args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_csv(args.corpus_output, index=False)
    pair_df.to_csv(args.pair_output, index=False)

    print(corpus_df.to_string(index=False))
    print(f"\nWrote corpus scores to {args.corpus_output}")
    print(f"Wrote pair scores to {args.pair_output}")


if __name__ == "__main__":
    main()
