#!/usr/bin/env python3
"""Run zero-shot direct SLM summarization via Ollama."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundlm.config import DEFAULT_SAMPLE_PATH, direct_model_configs
from groundlm.generation.direct import run_all_models, run_model
from groundlm.io import load_examples, write_json, write_results_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chinese summaries with open-weight SLMs (Ollama)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument(
        "--model",
        choices=["all", "qwen3.5", "gemma4", "aya_expanse"],
        default="all",
        help="Run one model or all three SLMs.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing checkpoints and outputs before running.",
    )
    parser.add_argument(
        "--combine-output",
        type=Path,
        default=Path("outputs/direct/all_models/direct_all_models_100samples_seed42_results.json"),
        help="Optional combined JSON path when --model=all.",
    )
    args = parser.parse_args()

    examples = load_examples(args.input)
    configs = direct_model_configs()

    if args.model == "all":
        results = run_all_models(args.input, reset=args.reset)
        combined: list[dict] = []
        for model_results in results.values():
            combined.extend(model_results)
        combined.sort(key=lambda row: (row["model_name"], int(row["sample_index"])))
        write_json(args.combine_output, combined)
        csv_path = args.combine_output.with_suffix(".csv")
        write_results_csv(csv_path, combined)
        print(f"Wrote combined results to {args.combine_output}")
        return

    config = next(cfg for cfg in configs if cfg.label == args.model)
    run_model(config, examples, reset=args.reset)
    print(f"Finished {config.label}. Results: {config.final_json_path}")


if __name__ == "__main__":
    main()
