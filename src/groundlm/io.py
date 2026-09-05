"""JSON/JSONL/CSV helpers for experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_processed_indices(path: Path) -> set[int]:
    return {
        int(record["sample_index"])
        for record in load_jsonl(path)
        if "sample_index" in record
    }


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples = load_json(path)
    if not isinstance(examples, list):
        raise TypeError(f"Expected a JSON array in {path}")
    return examples


def write_results_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "model_name",
        "generated_summary_zh",
        "reference_summary_zh",
        "reference_summary_en",
        "dialogue",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def records_to_results(
    records: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    """Normalize checkpoint records into the shared results schema."""
    results: list[dict[str, Any]] = []
    for record in records:
        results.append(
            {
                "sample_index": record.get("sample_index", record.get("id")),
                "model_name": model_name,
                "generated_summary_zh": record.get(
                    "generated_summary_zh", record.get("prediction", "")
                ).strip(),
                "reference_summary_zh": record.get(
                    "reference_summary_zh", record.get("reference", "")
                ),
                "reference_summary_en": record.get(
                    "reference_summary_en", record.get("summary", "")
                ),
                "dialogue": record.get("dialogue", ""),
            }
        )
    return sorted(results, key=lambda row: int(row["sample_index"]))
