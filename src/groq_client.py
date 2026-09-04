"""
groq_client.py — Groq LLM client wrapper
Never logs or exposes the API key.
"""

from typing import Iterator, Optional

from src import config
from src.utils import get_logger

logger = get_logger("groq_client")

_CLIENT = None


class ModelUnavailableError(RuntimeError):
    """Raised when a model outside the authoritative allow-list is requested."""


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or Streamlit secrets."
            )
        from groq import Groq
        _CLIENT = Groq(api_key=config.GROQ_API_KEY)
    return _CLIENT


def _resolve_model(model: Optional[str]) -> str:
    """
    Resolve and validate the model against the single authoritative allow-list
    (config.AVAILABLE_MODELS). There is NO silent fallback to another model: an
    unsupported model raises a clear operational error so a misconfiguration
    surfaces immediately instead of quietly answering with the wrong model.
    """
    mdl = (model or config.GROQ_MODEL or "").strip()
    if mdl not in config.AVAILABLE_MODELS:
        raise ModelUnavailableError(
            f"Requested model {mdl!r} is not a supported model. "
            f"Supported models: {config.AVAILABLE_MODELS}. "
            f"Set GROQ_MODEL to a supported value (currently {config.GROQ_MODEL!r})."
        )
    return mdl


def generate(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = config.DEFAULT_TEMP,
    max_tokens: int = 1500,
) -> str:
    """Single-turn generation; returns assistant text."""
    client = _get_client()
    mdl = _resolve_model(model)
    response = client.chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def stream(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = config.DEFAULT_TEMP,
    max_tokens: int = 1500,
) -> Iterator[str]:
    """Streaming generation; yields text delta strings."""
    client = _get_client()
    mdl = _resolve_model(model)
    with client.chat.completions.stream(
        model=mdl,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    ) as stream_ctx:
        for chunk in stream_ctx:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
