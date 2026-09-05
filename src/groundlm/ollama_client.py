"""Ollama HTTP client for local SLM inference."""

from __future__ import annotations

import requests

from groundlm.config import (
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_TEMPERATURE,
    OLLAMA_HOST,
)


def get_installed_models(host: str = OLLAMA_HOST) -> list[str]:
    response = requests.get(f"{host}/api/tags", timeout=10)
    response.raise_for_status()
    return [model.get("name", "") for model in response.json().get("models", [])]


def check_models_available(
    required: list[str],
    host: str = OLLAMA_HOST,
) -> list[str]:
    """Return model names that are required but not installed locally."""
    installed = set(get_installed_models(host))
    return [name for name in required if name not in installed]


def call_ollama(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    timeout: int = DEFAULT_OLLAMA_TIMEOUT,
    host: str = OLLAMA_HOST,
) -> str:
    """Call Ollama's chat API and return the assistant message content."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "think": False,
    }

    response = requests.post(
        f"{host}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    content = response.json().get("message", {}).get("content", "")
    return (content or "").strip()
