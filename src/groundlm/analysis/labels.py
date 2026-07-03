"""Error-label parsing and aggregation for the grounding taxonomy."""

from __future__ import annotations

import re

import pandas as pd

from groundlm.config import MODEL_ORDER, TAXONOMY_LABELS


def split_labels(value: str) -> list[str]:
    """Split a multi-label annotation string into individual labels."""
    if pd.isna(value) or not str(value).strip():
        return []
    parts = re.split(r"[|,;]+", str(value))
    return [part.strip() for part in parts if part.strip()]


def explode_labels(df: pd.DataFrame, label_col: str = "error_label") -> pd.DataFrame:
    working = df.copy()
    working["label_list"] = working[label_col].map(split_labels)
    return working.explode("label_list").rename(columns={"label_list": "label"})


def label_rates_by_model(
    df: pd.DataFrame,
    *,
    label_col: str = "error_label",
    model_col: str = "model_name",
    n_samples_per_model: int = 100,
) -> pd.DataFrame:
    long = explode_labels(df, label_col=label_col)
    counts = (
        pd.crosstab(long[model_col], long["label"])
        .reindex(index=MODEL_ORDER, columns=TAXONOMY_LABELS, fill_value=0)
    )
    return counts.div(n_samples_per_model)


def aggregated_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Table-2-style aggregated rates (hallucination, omission, language)."""
    rows: list[dict[str, float | str]] = []
    for model in MODEL_ORDER:
        part = df[df["model_name"] == model]
        if part.empty:
            continue
        labels = part["error_label"].map(split_labels)
        n = len(part)
        rows.append(
            {
                "model_name": model,
                "no_error": sum("NO-ERROR" in lbls and len(lbls) == 1 for lbls in labels) / n,
                "hallucination": sum(any(l.startswith("H-") for l in lbls) for lbls in labels) / n,
                "omission": sum("OMIT" in lbls for lbls in labels) / n,
                "language": sum("LANG" in lbls for lbls in labels) / n,
            }
        )
    return pd.DataFrame(rows)


def summary_length_ratio(prediction: str, reference: str) -> float | None:
    ref_len = sum(1 for char in reference if not char.isspace())
    if ref_len == 0:
        return None
    pred_len = sum(1 for char in prediction if not char.isspace())
    return pred_len / ref_len
