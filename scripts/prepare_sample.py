#!/usr/bin/env python3
"""Create the fixed 100-example evaluation subset (seed=42)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a reproducible subset from the XSAMSum test split."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/test.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/test_100_seed42.json"),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if args.sample_size > len(records):
        raise ValueError(
            f"sample-size={args.sample_size} exceeds input size={len(records)}"
        )

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(records)), args.sample_size))
    sampled = [records[i] for i in indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Wrote {len(sampled)} examples to {args.output} "
        f"(seed={args.seed})."
    )


if __name__ == "__main__":
    main()
