#!/usr/bin/env python3
"""Summarize grounding-error label statistics from annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundlm.analysis.labels import aggregated_error_rates, label_rates_by_model
from groundlm.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report taxonomy label statistics (Tables 1–2 style)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "annotations" / "qualitative_direct.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics",
    )
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fine_grained = label_rates_by_model(df)
    aggregated = aggregated_error_rates(df)

    fine_path = args.output_dir / "fine_grained_label_rates.csv"
    agg_path = args.output_dir / "aggregated_error_rates.csv"
    fine_grained.to_csv(fine_path)
    aggregated.to_csv(agg_path, index=False)

    print("=== Aggregated error rates ===")
    print(aggregated.to_string(index=False))
    print(f"\nWrote {fine_path}")
    print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
