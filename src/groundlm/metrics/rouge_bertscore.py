"""ROUGE and BERTScore evaluation following the ClidSum/XSAMSum protocol."""

from __future__ import annotations

from typing import Any

import pandas as pd
from bert_score import BERTScorer
from rouge_score import rouge_scorer

from groundlm.config import (
    BERTSCORE_LANG_ZH,
    BERTSCORE_MODEL_ZH,
    BERTSCORE_NUM_LAYERS,
    ROUGE_LANG_ZH,
    ROUGE_TYPES,
)


def compute_rouge(
    predictions: list[str],
    references: list[str],
    *,
    rouge_types: tuple[str, ...] = ROUGE_TYPES,
    language: str = ROUGE_LANG_ZH,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    scorer = rouge_scorer.RougeScorer(
        rouge_types=list(rouge_types),
        lang=language,
        use_stemmer=True,
    )

    pair_scores: list[dict[str, float]] = []
    aggregated = {rt: 0.0 for rt in rouge_types}

    for pred, ref in zip(predictions, references, strict=True):
        scores = scorer.score(ref, pred)
        pair = {rt: round(scores[rt].fmeasure * 100, 2) for rt in rouge_types}
        pair_scores.append(pair)
        for rt in rouge_types:
            aggregated[rt] += scores[rt].fmeasure

    n = len(predictions)
    corpus_scores = {rt: round(aggregated[rt] / n * 100, 2) for rt in rouge_types}
    return corpus_scores, pair_scores


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = BERTSCORE_MODEL_ZH,
    lang: str = BERTSCORE_LANG_ZH,
    num_layers: int = BERTSCORE_NUM_LAYERS,
) -> tuple[dict[str, float], list[float]]:
    scorer = BERTScorer(
        model_type=model_type,
        lang=lang,
        num_layers=num_layers,
        rescale_with_baseline=False,
    )
    _, _, f1 = scorer.score(predictions, references)
    pair_scores = [round(float(value) * 100, 2) for value in f1.tolist()]
    corpus_f1 = round(sum(pair_scores) / len(pair_scores), 2)
    return {"f1": corpus_f1}, pair_scores


def evaluate_dataframe(
    df: pd.DataFrame,
    *,
    pred_col: str = "generated_summary_zh",
    ref_col: str = "reference_summary_zh",
    model_col: str = "model_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute corpus- and pair-level ROUGE/BERTScore for each model."""
    corpus_rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []

    for model_name, group in df.groupby(model_col, sort=False):
        predictions = group[pred_col].astype(str).tolist()
        references = group[ref_col].astype(str).tolist()

        rouge_corpus, rouge_pairs = compute_rouge(predictions, references)
        bs_corpus, bs_pairs = compute_bertscore(predictions, references)

        corpus_row = {"model_name": model_name, **rouge_corpus, "bs_f1_raw": bs_corpus["f1"]}
        corpus_rows.append(corpus_row)

        pair_df = group.reset_index(drop=True).copy()
        for rt in ROUGE_TYPES:
            pair_df[rt] = [row[rt] for row in rouge_pairs]
        pair_df["bs_f1_raw"] = bs_pairs
        pair_frames.append(pair_df)

    return pd.DataFrame(corpus_rows), pd.concat(pair_frames, ignore_index=True)
