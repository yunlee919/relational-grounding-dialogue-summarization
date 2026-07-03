"""OmniScore evaluation for dialogue summarization."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from groundlm.config import OMNISCORE_MAX_LEN, OMNISCORE_REPO, OMNISCORE_TASK

SCORE_NAMES = ("informativeness", "clarity", "plausibility", "faithfulness")


def load_omniscore(
    repo_id: str = OMNISCORE_REPO,
    device: str | None = None,
) -> tuple[Any, Any, list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    score_names = list(getattr(model.config, "score_names", SCORE_NAMES))
    return model, tokenizer, score_names


def format_source_grounded(dialogue: str, candidate: str, task: str = OMNISCORE_TASK) -> str:
    return f"Task: {task}\nSource: {dialogue}\nCandidate: {candidate}"


def format_reference_based(
    reference: str,
    candidate: str,
    task: str = OMNISCORE_TASK,
) -> str:
    return f"Task: {task}\nReference: {reference}\nCandidate: {candidate}"


def score_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    *,
    max_len: int = OMNISCORE_MAX_LEN,
    batch_size: int = 8,
) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    all_scores: list[dict[str, float]] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
        predictions = outputs.predictions.detach().cpu().tolist()
        score_names = list(getattr(model.config, "score_names", SCORE_NAMES))
        for row in predictions:
            all_scores.append(
                {name: round(float(value), 4) for name, value in zip(score_names, row, strict=True)}
            )

    return all_scores


def evaluate_dataframe(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    *,
    formatter: Callable[[pd.Series], str],
    mode_name: str,
    batch_size: int = 8,
) -> pd.DataFrame:
    texts = [formatter(row) for _, row in df.iterrows()]
    scores: list[dict[str, float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"OmniScore ({mode_name})"):
        chunk = texts[start : start + batch_size]
        scores.extend(score_texts(chunk, model, tokenizer, batch_size=len(chunk)))

    result = df.reset_index(drop=True).copy()
    for name in SCORE_NAMES:
        result[name] = [row.get(name) for row in scores]
    result["mode"] = mode_name
    return result
