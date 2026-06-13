#!/usr/bin/env python3
"""Run the mBART English-dialogue-to-Chinese-summary baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL = "jjnhuang/mbart-large-50-en-dialogue-to-zh-summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese summaries for a sampled test JSON file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/splits/test_100_seed42.json"),
        help="Input JSON file containing test examples.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/mbart_baseline/"
            "mbart_large50_100samples_seed42_results.json"
        ),
        help="Path for JSON results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/mbart_baseline/"
            "mbart_large50_100samples_seed42_results.csv"
        ),
        help="Path for CSV results.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--max-source-tokens",
        type=int,
        default=1024,
        help="Maximum input token length.",
    )
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=128,
        help="Maximum generated summary length.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam size for generation.",
    )
    parser.add_argument(
        "--source-lang",
        default="en_XX",
        help="mBART source language code.",
    )
    parser.add_argument(
        "--target-lang",
        default="zh_CN",
        help="mBART target language code.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to use for inference.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batched(records: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


def get_forced_bos_token_id(tokenizer: Any, target_lang: str) -> int | None:
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and target_lang in lang_code_to_id:
        return lang_code_to_id[target_lang]

    token_id = tokenizer.convert_tokens_to_ids(target_lang)
    if token_id != tokenizer.unk_token_id:
        return token_id

    return None


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "model_name",
        "generated_summary_zh",
        "reference_summary_zh",
        "reference_summary_en",
        "dialogue",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    with args.input.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    if not isinstance(examples, list):
        raise TypeError(f"Expected a top-level JSON array in {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = args.source_lang

    forced_bos_token_id = get_forced_bos_token_id(tokenizer, args.target_lang)
    generation_kwargs: dict[str, Any] = {
        "max_length": args.max_target_tokens,
        "num_beams": args.num_beams,
    }
    if forced_bos_token_id is not None:
        generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(examples), desc="Generating summaries")

    for start, batch in batched(examples, args.batch_size):
        dialogues = [example["dialogue"] for example in batch]
        encoded = tokenizer(
            dialogues,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**encoded, **generation_kwargs)

        generated_texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for offset, (example, generated_summary) in enumerate(
            zip(batch, generated_texts, strict=True)
        ):
            results.append(
                {
                    "sample_index": start + offset,
                    "model_name": args.model_name,
                    "generated_summary_zh": generated_summary.strip(),
                    "reference_summary_zh": example.get("summary_zh", ""),
                    "reference_summary_en": example.get("summary", ""),
                    "dialogue": example.get("dialogue", ""),
                }
            )

        progress.update(len(batch))

    progress.close()
    write_json(args.output_json, results)
    write_csv(args.output_csv, results)
    print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote CSV results to {args.output_csv}")


if __name__ == "__main__":
    main()
