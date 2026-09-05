"""Generate paper figures from annotation data."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from groundlm.analysis.labels import explode_labels, split_labels
from groundlm.config import MODEL_DISPLAY, MODEL_ORDER, ROOT, TAXONOMY_LABELS

MODEL_LABELS = [MODEL_DISPLAY[name] for name in MODEL_ORDER]
LENGTH_BIN_ORDER = ["<0.75", "0.75-1.25", "1.25-2.0", "2.0-4.0", ">4.0"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["label_list"] = working["error_label"].map(split_labels)
    working["is_no_error"] = working["error_label"].eq("NO-ERROR")
    working["has_error"] = ~working["is_no_error"]
    working["rouge_mean"] = working[["rouge1", "rouge2", "rougeL"]].mean(axis=1)
    working["length_ratio_bin"] = pd.cut(
        working["Length Ratio"],
        bins=[-np.inf, 0.75, 1.25, 2.0, 4.0, np.inf],
        labels=LENGTH_BIN_ORDER,
        right=False,
    )
    return working


def export_appendix_figures(
    annotation_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    df = _prepare(pd.read_csv(annotation_path))
    label_long = explode_labels(df).rename(columns={"label_list": "label"})

    model_label_rates = (
        pd.crosstab(label_long["model_name"], label_long["label"])
        .reindex(index=MODEL_ORDER, columns=TAXONOMY_LABELS, fill_value=0)
        .div(100)
    )

    no_error_wide = (
        df.pivot(index="sample_index", columns="model_name", values="is_no_error")
        .reindex(columns=MODEL_ORDER)
    )
    pairwise_wins = pd.DataFrame(0, index=MODEL_ORDER, columns=MODEL_ORDER, dtype=int)
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        pairwise_wins.loc[model_a, model_b] = int(
            (no_error_wide[model_a] & ~no_error_wide[model_b]).sum()
        )
        pairwise_wins.loc[model_b, model_a] = int(
            (no_error_wide[model_b] & ~no_error_wide[model_a]).sum()
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1.45, 1]})
    sns.heatmap(model_label_rates, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=0.65, ax=axes[0])
    axes[0].set(title="(a) Fine-grained label rates", xlabel="Label", ylabel="Model")
    axes[0].set_yticklabels(MODEL_LABELS, rotation=0)
    sns.heatmap(pairwise_wins, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1])
    axes[1].set(title="(b) Pairwise no-error wins", xlabel="Losing model", ylabel="Winning model")
    axes[1].set_xticklabels(MODEL_LABELS, rotation=35, ha="right")
    axes[1].set_yticklabels(MODEL_LABELS, rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "model_error_and_pairwise.pdf", bbox_inches="tight")
    plt.close(fig)

    # Length vs label rates
    bin_sizes = df["length_ratio_bin"].value_counts().reindex(LENGTH_BIN_ORDER, fill_value=0)
    bin_label_rates = (
        pd.crosstab(label_long["length_ratio_bin"], label_long["label"])
        .reindex(index=LENGTH_BIN_ORDER, columns=TAXONOMY_LABELS, fill_value=0)
        .div(bin_sizes, axis=0)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.heatmap(bin_label_rates.T, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=0.65, ax=ax)
    ax.set(title="Label rates by length ratio bin", xlabel="Length ratio bin", ylabel="Label")
    fig.tight_layout()
    fig.savefig(output_dir / "length_label_rates.pdf", bbox_inches="tight")
    plt.close(fig)

    # Multi-label co-occurrence among hallucination labels
    halluc_labels = [label for label in TAXONOMY_LABELS if label.startswith("H-")]
    cooc = pd.DataFrame(0, index=halluc_labels, columns=halluc_labels, dtype=int)
    for labels in df["label_list"]:
        present = [label for label in labels if label in halluc_labels]
        for a, b in combinations(sorted(set(present)), 2):
            cooc.loc[a, b] += 1
            cooc.loc[b, a] += 1

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(cooc, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set(title="Hallucination label co-occurrence")
    fig.tight_layout()
    fig.savefig(output_dir / "multilabel_cooccurrence.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    annotation_path = ROOT / "annotations" / "qualitative_direct.csv"
    output_dir = ROOT / "figures" / "appendix"
    export_appendix_figures(annotation_path, output_dir)
    print(f"Wrote appendix figures to {output_dir}")


if __name__ == "__main__":
    main()
