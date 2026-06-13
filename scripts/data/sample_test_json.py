#!/usr/bin/env python3
"""Sample a fixed-size subset from data/raw/test.json."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible random sample from a JSON array."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/test.json"),
        help="Input JSON file containing a top-level array.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/test_100_seed42.json"),
        help="Output path for the sampled JSON array.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to draw without replacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise TypeError(f"Expected a top-level JSON array in {args.input}")

    if args.sample_size > len(records):
        raise ValueError(
            f"sample-size={args.sample_size} exceeds input size={len(records)}"
        )

    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(records)), args.sample_size))
    sampled_records = [records[index] for index in sampled_indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(sampled_records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Wrote {len(sampled_records)} samples from {args.input} to {args.output} "
        f"(seed={args.seed})."
    )


if __name__ == "__main__":
    main()
