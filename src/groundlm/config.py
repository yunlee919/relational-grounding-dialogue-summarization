"""Shared configuration for GroundLM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up from *start*."""
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "data" / "splits" / "test_100_seed42.json").exists():
            return path
    raise FileNotFoundError(
        "Could not locate project root (expected data/splits/test_100_seed42.json)."
    )


ROOT = project_root()

# --- Data ---
DEFAULT_SAMPLE_PATH = ROOT / "data" / "splits" / "test_100_seed42.json"
SAMPLE_SIZE = 100
SAMPLE_SEED = 42

# --- Ollama / SLM inference ---
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 1024
DEFAULT_OLLAMA_TIMEOUT = 900
SLEEP_SECONDS = 0.5

# --- mBART baseline ---
MBART_MODEL = "jjnhuang/mbart-large-50-en-dialogue-to-zh-summary"
MBART_SOURCE_LANG = "en_XX"
MBART_TARGET_LANG = "zh_CN"

# --- Evaluation ---
ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
ROUGE_LANG_ZH = "chinese"
BERTSCORE_MODEL_ZH = "hfl/chinese-bert-wwm-ext"
BERTSCORE_LANG_ZH = "zh"
BERTSCORE_NUM_LAYERS = 8
OMNISCORE_REPO = "QCRI/OmniScore-deberta-v3"
OMNISCORE_MAX_LEN = 512
OMNISCORE_TASK = "summarization"

# --- Taxonomy ---
TAXONOMY_LABELS = (
    "H-ENT",
    "H-EVT",
    "H-ROLE",
    "H-CIRC",
    "H-MOD",
    "H-DISC",
    "OMIT",
    "LANG",
    "NO-ERROR",
)

MODEL_ORDER = ("aya-expanse:8b", "gemma4:e4b", "qwen3.5:9b", "mBART-large")
MODEL_DISPLAY = {
    "aya-expanse:8b": "Aya Expanse 8B",
    "gemma4:e4b": "Gemma 4 E4B",
    "qwen3.5:9b": "Qwen 3.5 9B",
    "mBART-large": "mBART-large",
}


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one zero-shot direct SLM."""

    label: str
    model_name: str
    file_prefix: str
    output_dir: Path

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / f"{self.file_prefix}_checkpoint.jsonl"

    @property
    def final_json_path(self) -> Path:
        return self.output_dir / f"{self.file_prefix}_results.json"

    @property
    def final_csv_path(self) -> Path:
        return self.output_dir / f"{self.file_prefix}_results.csv"

    @property
    def error_path(self) -> Path:
        return self.output_dir / f"{self.file_prefix}_errors.jsonl"


def direct_model_configs(root: Path | None = None) -> list[ModelConfig]:
    """Return the three SLMs evaluated in the paper."""
    base = (root or ROOT) / "outputs" / "direct"
    return [
        ModelConfig(
            label="qwen3.5",
            model_name="qwen3.5:9b",
            file_prefix="direct_qwen9b_100samples_seed42",
            output_dir=base / "qwen3.5",
        ),
        ModelConfig(
            label="gemma4",
            model_name="gemma4:e4b",
            file_prefix="direct_gemma4_e4b_100samples_seed42",
            output_dir=base / "gemma4",
        ),
        ModelConfig(
            label="aya_expanse",
            model_name="aya-expanse:8b",
            file_prefix="direct_aya8b_100samples_seed42",
            output_dir=base / "aya_expanse",
        ),
    ]
