#!/usr/bin/env python3
"""Add Length Ratio = prediction length / reference length to a CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def summary_length(text: str, unit: str) -> int:
    if unit == "chars":
        return sum(1 for char in text if not char.isspace())
    if unit == "words":
        return len(text.split())
    raise ValueError(f"Unsupported length unit: {unit}")


def format_ratio(prediction: str, reference: str, unit: str) -> str:
    reference_length = summary_length(reference, unit)
    if reference_length == 0:
        return ""
    prediction_length = summary_length(prediction, unit)
    return f"{prediction_length / reference_length:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Length Ratio column to a CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--ratio-column", default="Length Ratio")
    parser.add_argument("--unit", choices=("chars", "words"), default="chars")
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{args.csv_path} has no header row")
        fieldnames = list(reader.fieldnames)
        if args.ratio_column not in fieldnames:
            fieldnames.append(args.ratio_column)
        rows = []
        for row in reader:
            row[args.ratio_column] = format_ratio(
                row.get(args.prediction_column, ""),
                row.get(args.reference_column, ""),
                args.unit,
            )
            rows.append(row)

    target = args.output or args.csv_path
    with target.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
