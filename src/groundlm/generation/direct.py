"""Zero-shot direct SLM summarization via Ollama."""

from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from groundlm.config import (
    SLEEP_SECONDS,
    ModelConfig,
    direct_model_configs,
)
from groundlm.io import (
    append_jsonl,
    load_examples,
    load_processed_indices,
    load_jsonl,
    records_to_results,
    write_json,
    write_results_csv,
)
from groundlm.ollama_client import call_ollama, check_models_available
from groundlm.prompts import fill_prompt, load_direct_prompt


def generate_summary(dialogue: str, config: ModelConfig, prompt_template: str) -> str:
    prompt = fill_prompt(prompt_template, {"dialogue": dialogue})
    return call_ollama(model=config.model_name, prompt=prompt)


def run_example(
    example: dict[str, Any],
    sample_index: int,
    config: ModelConfig,
    prompt_template: str,
) -> dict[str, Any]:
    dialogue = example.get("dialogue", "")
    generated = generate_summary(dialogue, config, prompt_template)
    return {
        "sample_index": sample_index,
        "model_name": config.model_name,
        "generated_summary_zh": generated,
        "reference_summary_zh": example.get("summary_zh", ""),
        "reference_summary_en": example.get("summary", ""),
        "dialogue": dialogue,
    }


def run_model(
    config: ModelConfig,
    examples: list[dict[str, Any]],
    *,
    prompt_template: str | None = None,
    reset: bool = False,
    sleep_seconds: float = SLEEP_SECONDS,
) -> list[dict[str, Any]]:
    """Run direct summarization for one model with checkpoint/resume support."""
    template = prompt_template or load_direct_prompt()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        for path in (
            config.checkpoint_path,
            config.final_json_path,
            config.final_csv_path,
            config.error_path,
        ):
            path.unlink(missing_ok=True)

    processed = load_processed_indices(config.checkpoint_path)
    progress = tqdm(total=len(examples), desc=config.label, initial=len(processed))

    for index, example in enumerate(examples):
        if index in processed:
            continue
        try:
            record = run_example(example, index, config, template)
            append_jsonl(record, config.checkpoint_path)
            processed.add(index)
        except Exception as exc:  # noqa: BLE001 — log and continue on transient API errors
            append_jsonl(
                {"sample_index": index, "error": str(exc)},
                config.error_path,
            )
        progress.update(1)
        time.sleep(sleep_seconds)

    progress.close()

    records = load_jsonl(config.checkpoint_path)
    results = records_to_results(records, config.model_name)
    write_json(config.final_json_path, results)
    write_results_csv(config.final_csv_path, results)
    return results


def run_all_models(
    input_path,
    *,
    models: list[ModelConfig] | None = None,
    reset: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Run all configured SLMs on the evaluation subset."""
    examples = load_examples(input_path)
    configs = models or direct_model_configs()

    missing = check_models_available([cfg.model_name for cfg in configs])
    if missing:
        raise RuntimeError(
            "Required Ollama models are not installed: "
            + ", ".join(missing)
            + ". Pull them with `ollama pull <model>`."
        )

    all_results: dict[str, list[dict[str, Any]]] = {}
    for config in configs:
        all_results[config.label] = run_model(config, examples, reset=reset)
    return all_results
