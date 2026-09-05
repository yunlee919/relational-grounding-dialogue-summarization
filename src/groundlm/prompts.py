"""Prompt templates for direct English-to-Chinese summarization."""

from __future__ import annotations

from pathlib import Path

from groundlm.config import ROOT

DIRECT_PROMPT = """You are an expert English-Chinese bilingual speaker. Given an English dialogue, please write a concise Simplified Chinese summary. Output only the summary, without titles or markdown.

English dialogue:
{dialogue}

Chinese summary:"""


def load_direct_prompt(path: Path | None = None) -> str:
    """Load the direct prompt template from disk or return the built-in default."""
    prompt_path = path or (ROOT / "prompts" / "direct_prompt.txt")
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DIRECT_PROMPT


def fill_prompt(template: str, replacements: dict[str, str]) -> str:
    """Replace named placeholders in a prompt template."""
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt
