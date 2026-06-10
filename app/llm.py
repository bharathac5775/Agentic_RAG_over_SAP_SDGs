"""
Provider-agnostic LLM facade.

Two public functions used everywhere in the project:
    chat(model, messages, *, format=None, temperature=0.0, num_ctx=NUM_CTX)
        -> str (the assistant message content)
    embed(model, text)
        -> list[float] (the embedding vector)

Provider is selected at call-time from environment variables:
    LLM_PROVIDER    one of: ollama (default), openai, anthropic, google
    EMBED_PROVIDER  same set; defaults to LLM_PROVIDER if unset

API keys come from env vars, never code:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GOOGLE_API_KEY
(Ollama needs no key; it talks to localhost:11434.)

Cloud SDKs are imported LAZILY inside their adapter, so installing the
project doesn't drag in `openai`, `anthropic`, or `google-generativeai`
unless you actually use them.

Why this exists:
    The brief says "Any LLM provider you have access to. Keep API keys out
    of the repo." This module makes provider switching a 1-3 env-var
    change with no code edits — and ensures keys live ONLY in env vars
    and a gitignored .env file.

How callers use it:
    from app import llm
    text = llm.chat(model="llama3.1:8b", messages=[...], format="json")
    vec  = llm.embed(model="nomic-embed-text", text="hello")

Each adapter returns plain Python (str / list[float]) so callers don't
need to dig through provider-specific response objects.
"""

from __future__ import annotations

import os
from typing import Any

from app import config


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").lower().strip()


def _embed_provider() -> str:
    return os.getenv("EMBED_PROVIDER", _llm_provider()).lower().strip()


def _require_env(var: str, provider: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(
            f"{provider} provider requires {var} to be set. "
            f"Put it in a .env file (gitignored) or export it in your shell. "
            f"Never commit it to the repo."
        )
    return val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    format: str | None = None,
    temperature: float = 0.0,
    num_ctx: int | None = None,
) -> str:
    """Send a chat completion to the configured provider; return the
    assistant message content as a string.

    Args:
        model:        Provider-specific model name (e.g. "llama3.1:8b",
                      "gpt-4o-mini", "claude-haiku-4-5", "gemini-2.0-flash").
        messages:     OpenAI-style list of {"role": "...", "content": "..."}.
                      "system" / "user" / "assistant" roles supported.
        format:       Pass "json" to request JSON-mode output. Adapter-specific.
        temperature:  Sampling temperature.
        num_ctx:      Context-window cap. Only the Ollama adapter respects it
                      (other providers manage context themselves).
    """
    if num_ctx is None:
        num_ctx = config.NUM_CTX

    provider = _llm_provider()
    if provider == "ollama":
        return _chat_ollama(model, messages, format, temperature, num_ctx)
    if provider == "openai":
        return _chat_openai(model, messages, format, temperature)
    if provider == "anthropic":
        return _chat_anthropic(model, messages, format, temperature)
    if provider == "google":
        return _chat_google(model, messages, format, temperature)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        f"Use one of: ollama, openai, anthropic, google."
    )


def embed(model: str, text: str) -> list[float]:
    """Embed `text` via the configured embedding provider. Returns a list
    of floats (vector).
    """
    provider = _embed_provider()
    if provider == "ollama":
        return _embed_ollama(model, text)
    if provider == "openai":
        return _embed_openai(model, text)
    if provider == "google":
        return _embed_google(model, text)
    if provider == "anthropic":
        # Anthropic does not currently offer a public embeddings API.
        raise RuntimeError(
            "Anthropic does not provide embeddings. Use EMBED_PROVIDER=ollama "
            "or EMBED_PROVIDER=openai for embeddings, and LLM_PROVIDER=anthropic "
            "for chat."
        )
    raise RuntimeError(
        f"Unknown EMBED_PROVIDER={provider!r}. Use one of: ollama, openai, google."
    )


# ===========================================================================
# Ollama adapter (default — local, no API key)
# ===========================================================================


def _chat_ollama(
    model: str,
    messages: list[dict[str, str]],
    format: str | None,
    temperature: float,
    num_ctx: int,
) -> str:
    import ollama  # always available — listed in requirements.txt

    options: dict[str, Any] = {"temperature": temperature, "num_ctx": num_ctx}
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "options": options}
    if format == "json":
        kwargs["format"] = "json"
    resp = ollama.chat(**kwargs)
    return resp["message"]["content"]


def _embed_ollama(model: str, text: str) -> list[float]:
    import ollama
    return ollama.embeddings(model=model, prompt=text)["embedding"]


# ===========================================================================
# OpenAI adapter
# ===========================================================================


def _chat_openai(
    model: str,
    messages: list[dict[str, str]],
    format: str | None,
    temperature: float,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "OpenAI provider requested but `openai` package is not installed. "
            "Run: pip install openai"
        ) from e

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY", "openai"))
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if format == "json":
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _embed_openai(model: str, text: str) -> list[float]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "OpenAI embeddings requested but `openai` package is not installed. "
            "Run: pip install openai"
        ) from e

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY", "openai"))
    resp = client.embeddings.create(model=model, input=text)
    return list(resp.data[0].embedding)


# ===========================================================================
# Anthropic adapter (Claude)
# ===========================================================================


def _chat_anthropic(
    model: str,
    messages: list[dict[str, str]],
    format: str | None,
    temperature: float,
) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "Anthropic provider requested but `anthropic` package is not installed. "
            "Run: pip install anthropic"
        ) from e

    # Anthropic separates `system` from `messages`. Extract any system role(s).
    system_parts: list[str] = []
    user_messages: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})

    # Anthropic doesn't have a JSON-mode flag like OpenAI's response_format.
    # The standard pattern is to instruct it via the system prompt.
    system = "\n\n".join(system_parts)
    if format == "json":
        system = (system + "\n\n" if system else "") + (
            "Return ONLY a single valid JSON object. No prose, no markdown, "
            "no code fences."
        )

    client = Anthropic(api_key=_require_env("ANTHROPIC_API_KEY", "anthropic"))
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system or None,
        messages=user_messages,
        temperature=temperature,
    )
    # Concatenate any text blocks. Tool-use blocks are not expected here.
    parts: list[str] = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


# ===========================================================================
# Google (Gemini) adapter
# ===========================================================================


def _chat_google(
    model: str,
    messages: list[dict[str, str]],
    format: str | None,
    temperature: float,
) -> str:
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "Google provider requested but `google-generativeai` package is not installed. "
            "Run: pip install google-generativeai"
        ) from e

    genai.configure(api_key=_require_env("GOOGLE_API_KEY", "google"))

    # Gemini takes `system_instruction` separately; convert remaining messages
    # to its content list (it uses "model" instead of "assistant").
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [m["content"]]})

    generation_config: dict[str, Any] = {"temperature": temperature}
    if format == "json":
        generation_config["response_mime_type"] = "application/json"

    gm = genai.GenerativeModel(
        model_name=model,
        system_instruction="\n\n".join(system_parts) or None,
    )
    resp = gm.generate_content(contents, generation_config=generation_config)
    return resp.text or ""


def _embed_google(model: str, text: str) -> list[float]:
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "Google embeddings requested but `google-generativeai` package is not installed. "
            "Run: pip install google-generativeai"
        ) from e

    genai.configure(api_key=_require_env("GOOGLE_API_KEY", "google"))
    resp = genai.embed_content(model=model, content=text)
    return list(resp["embedding"])


# ---------------------------------------------------------------------------
# Tiny self-check helper used by /health and the smoke script
# ---------------------------------------------------------------------------


def provider_summary() -> dict[str, str]:
    """Return a structured description of which providers are active.
    Safe to expose in /health responses — does not include secrets.
    """
    return {
        "llm_provider": _llm_provider(),
        "embed_provider": _embed_provider(),
    }
