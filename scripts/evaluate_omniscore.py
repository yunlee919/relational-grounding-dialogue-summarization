#!/usr/bin/env python3
"""Compute OmniScore for model outputs (reference-based mode used in the paper)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from groundlm.metrics.omniscore import (
    evaluate_dataframe,
    format_reference_based,
    format_source_grounded,
    load_omniscore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate summaries with OmniScore.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/direct/all_models/direct_all_models_100samples_seed42_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/omniscore_pair_scores.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/metrics/omniscore_summary.csv"),
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=Path("outputs/metrics/omniscore_run_meta.json"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=["reference_based", "source_grounded", "both"],
        default="reference_based",
        help="OmniScore input mode (paper uses reference_based).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model, tokenizer, score_names = load_omniscore()

    frames: list[pd.DataFrame] = []
    modes: list[tuple[str, object]] = []
    if args.mode in ("reference_based", "both"):
        modes.append(
            (
                "reference_based",
                lambda row: format_reference_based(
                    row["reference_summary_zh"], row["generated_summary_zh"]
                ),
            )
        )
    if args.mode in ("source_grounded", "both"):
        modes.append(
            (
                "source_grounded",
                lambda row: format_source_grounded(
                    row["dialogue"], row["generated_summary_zh"]
                ),
            )
        )

    for model_name, group in df.groupby("model_name", sort=False):
        for mode_name, formatter in modes:
            scored = evaluate_dataframe(
                group,
                model,
                tokenizer,
                formatter=formatter,
                mode_name=mode_name,
                batch_size=args.batch_size,
            )
            scored["model_name"] = model_name
            frames.append(scored)

    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = (
        result.groupby(["model_name", "mode"])[score_names]
        .mean()
        .reset_index()
        .round(3)
    )
    summary.to_csv(args.summary_output, index=False)

    meta = {
        "input": str(args.input),
        "modes": [mode for mode, _ in modes],
        "score_names": score_names,
    }
    args.meta_output.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nWrote pair scores to {args.output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
