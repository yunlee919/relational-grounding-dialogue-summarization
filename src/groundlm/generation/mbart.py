"""mBART supervised baseline for English-to-Chinese dialogue summarization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from groundlm.config import (
    MBART_MODEL,
    MBART_SOURCE_LANG,
    MBART_TARGET_LANG,
)
from groundlm.io import load_examples, write_json, write_results_csv


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_forced_bos_token_id(tokenizer: Any, target_lang: str) -> int | None:
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and target_lang in lang_code_to_id:
        return lang_code_to_id[target_lang]
    token_id = tokenizer.convert_tokens_to_ids(target_lang)
    if token_id != tokenizer.unk_token_id:
        return token_id
    return None


def generate_mbart_summaries(
    examples: list[dict[str, Any]],
    *,
    model_name: str = MBART_MODEL,
    batch_size: int = 2,
    max_source_tokens: int = 1024,
    max_target_tokens: int = 128,
    num_beams: int = 4,
    source_lang: str = MBART_SOURCE_LANG,
    target_lang: str = MBART_TARGET_LANG,
    device: str = "auto",
) -> list[dict[str, Any]]:
    torch_device = choose_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)
    model.eval()

    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang

    forced_bos_token_id = get_forced_bos_token_id(tokenizer, target_lang)
    generation_kwargs: dict[str, Any] = {
        "max_length": max_target_tokens,
        "num_beams": num_beams,
    }
    if forced_bos_token_id is not None:
        generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(examples), desc="mBART")

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        dialogues = [example["dialogue"] for example in batch]
        encoded = tokenizer(
            dialogues,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_tokens,
        )
        encoded = {key: value.to(torch_device) for key, value in encoded.items()}

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
                    "model_name": model_name,
                    "generated_summary_zh": generated_summary.strip(),
                    "reference_summary_zh": example.get("summary_zh", ""),
                    "reference_summary_en": example.get("summary", ""),
                    "dialogue": example.get("dialogue", ""),
                }
            )
        progress.update(len(batch))

    progress.close()
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chinese summaries with the mBART baseline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/splits/test_100_seed42.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/mbart_baseline/mbart_large50_100samples_seed42_results.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/mbart_baseline/mbart_large50_100samples_seed42_results.csv"),
    )
    parser.add_argument("--model-name", default=MBART_MODEL)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-source-tokens", type=int, default=1024)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--source-lang", default=MBART_SOURCE_LANG)
    parser.add_argument("--target-lang", default=MBART_TARGET_LANG)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args(argv)

    examples = load_examples(args.input)
    results = generate_mbart_summaries(
        examples,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_source_tokens=args.max_source_tokens,
        max_target_tokens=args.max_target_tokens,
        num_beams=args.num_beams,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
    )

    write_json(args.output_json, results)
    write_results_csv(args.output_csv, results)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
