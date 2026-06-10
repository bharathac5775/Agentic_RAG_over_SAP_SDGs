"""Tests for the LLM provider abstraction.

Verifies provider dispatch, env-var-based selection, and helpful errors
when cloud SDKs aren't installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app import llm


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


class TestChatDispatch:

    def test_default_provider_is_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with patch("app.llm._chat_ollama", return_value="hi") as m:
            out = llm.chat("model", [{"role": "user", "content": "x"}])
        m.assert_called_once()
        assert out == "hi"

    def test_explicit_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        with patch("app.llm._chat_ollama", return_value="hi") as m:
            out = llm.chat("model", [{"role": "user", "content": "x"}])
        m.assert_called_once()
        assert out == "hi"

    def test_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with patch("app.llm._chat_openai", return_value="hi") as m:
            out = llm.chat("model", [{"role": "user", "content": "x"}])
        m.assert_called_once()
        assert out == "hi"

    def test_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        with patch("app.llm._chat_anthropic", return_value="hi") as m:
            out = llm.chat("model", [{"role": "user", "content": "x"}])
        m.assert_called_once()
        assert out == "hi"

    def test_google(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "google")
        with patch("app.llm._chat_google", return_value="hi") as m:
            out = llm.chat("model", [{"role": "user", "content": "x"}])
        m.assert_called_once()
        assert out == "hi"

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "wibble")
        with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
            llm.chat("m", [])


class TestEmbedDispatch:

    def test_default_embed_provider_is_ollama(self, monkeypatch):
        monkeypatch.delenv("EMBED_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with patch("app.llm._embed_ollama", return_value=[0.1] * 8) as m:
            out = llm.embed("m", "hi")
        m.assert_called_once()
        assert isinstance(out, list) and len(out) == 8

    def test_embed_provider_overrides_llm_provider(self, monkeypatch):
        # Generate via openai, but embed locally via ollama
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("EMBED_PROVIDER", "ollama")
        with patch("app.llm._embed_ollama", return_value=[0.0]) as m_ollama, \
             patch("app.llm._embed_openai") as m_openai:
            llm.embed("m", "hi")
        m_ollama.assert_called_once()
        m_openai.assert_not_called()

    def test_anthropic_embed_raises_helpful_error(self, monkeypatch):
        monkeypatch.setenv("EMBED_PROVIDER", "anthropic")
        with pytest.raises(RuntimeError, match="Anthropic does not provide embeddings"):
            llm.embed("m", "hi")


# ---------------------------------------------------------------------------
# Lazy-import behavior
# ---------------------------------------------------------------------------


class TestLazyImports:

    def test_openai_chat_without_package_raises_clearly(self, monkeypatch):
        """If `openai` isn't installed, calling chat() with LLM_PROVIDER=openai
        should raise a RuntimeError with installation hint, not ImportError.
        """
        # Simulate the package being missing by patching __import__
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="pip install openai"):
            llm.chat("gpt-4o-mini", [{"role": "user", "content": "x"}])

    def test_anthropic_chat_without_package_raises_clearly(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="pip install anthropic"):
            llm.chat("claude-haiku-4-5", [{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------


class TestKeyHandling:

    def test_openai_missing_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # We need openai-the-package importable to reach the key check.
        # Patch the OpenAI client constructor to a no-op so we don't actually
        # need it installed; the _require_env check fires before construction.
        fake_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": fake_openai}):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                llm.chat("gpt-4o-mini", [{"role": "user", "content": "x"}])

    def test_anthropic_missing_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        fake = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake}):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                llm.chat("claude-haiku-4-5", [{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# provider_summary
# ---------------------------------------------------------------------------


class TestProviderSummary:

    def test_summary_reflects_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("EMBED_PROVIDER", "ollama")
        s = llm.provider_summary()
        assert s == {"llm_provider": "openai", "embed_provider": "ollama"}

    def test_summary_does_not_include_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-key-do-not-leak")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-leak")
        s = llm.provider_summary()
        # Defense-in-depth: the dict must contain only provider names, no values
        # from API_KEY env vars.
        for v in s.values():
            assert "sk-" not in v
            assert "ant-" not in v
