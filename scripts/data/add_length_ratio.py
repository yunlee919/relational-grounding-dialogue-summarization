#!/usr/bin/env python3
"""Add a Length Ratio column to a summary evaluation CSV.

Length Ratio = predicted summary length / reference summary length.

The default length unit is non-whitespace characters, which is usually more
appropriate than whitespace-delimited words for Chinese summaries.
"""

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


def add_length_ratio(
    csv_path: Path,
    output_path: Path | None,
    prediction_column: str,
    reference_column: str,
    ratio_column: str,
    unit: str,
) -> None:
    with csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")

        missing_columns = [
            column
            for column in (prediction_column, reference_column)
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

        fieldnames = list(reader.fieldnames)
        if ratio_column not in fieldnames:
            fieldnames.append(ratio_column)

        rows = []
        for row in reader:
            row[ratio_column] = format_ratio(
                row.get(prediction_column, ""),
                row.get(reference_column, ""),
                unit,
            )
            rows.append(row)

    target_path = output_path or csv_path
    with target_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Length Ratio = prediction length / reference length to a CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. Defaults to updating the input file in place.",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Prediction summary column name. Defaults to 'prediction'.",
    )
    parser.add_argument(
        "--reference-column",
        default="reference",
        help="Reference summary column name. Defaults to 'reference'.",
    )
    parser.add_argument(
        "--ratio-column",
        default="Length Ratio",
        help="Output ratio column name. Defaults to 'Length Ratio'.",
    )
    parser.add_argument(
        "--unit",
        choices=("chars", "words"),
        default="chars",
        help="Length unit. Defaults to non-whitespace characters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_length_ratio(
        csv_path=args.csv_path,
        output_path=args.output,
        prediction_column=args.prediction_column,
        reference_column=args.reference_column,
        ratio_column=args.ratio_column,
        unit=args.unit,
    )


if __name__ == "__main__":
    main()
