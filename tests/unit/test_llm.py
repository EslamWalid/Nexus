from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.llm import base


def test_llm_requires_primary_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY_LLM1", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_LLM2", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY_LLM1 is missing"):
        base.GeminiLLM()


@patch("nexus.llm.base.ChatGoogleGenerativeAI")
def test_llm_initialization_without_fallback(mock_chat, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_LLM1", "test-key-1")
    monkeypatch.delenv("GEMINI_API_KEY_LLM2", raising=False)

    llm = base.GeminiLLM()

    mock_chat.assert_called_once()

    assert llm.primary_llm is not None
    assert llm.llm is llm.primary_llm


@patch("nexus.llm.base.ChatGoogleGenerativeAI")
def test_llm_initialization_with_fallback(mock_chat, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_LLM1", "test-key-1")
    monkeypatch.setenv("GEMINI_API_KEY_LLM2", "test-key-2")

    primary = MagicMock()
    fallback = MagicMock()

    primary.with_fallbacks.return_value = MagicMock()

    mock_chat.side_effect = [primary, fallback]

    llm = base.GeminiLLM()

    assert mock_chat.call_count == 2
    primary.with_fallbacks.assert_called_once_with([fallback])
    assert llm.llm is primary.with_fallbacks.return_value


@pytest.mark.asyncio
@patch("nexus.llm.base.ChatGoogleGenerativeAI")
async def test_llm_ainvoke(mock_chat, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_LLM1", "test-key-1")
    monkeypatch.delenv("GEMINI_API_KEY_LLM2", raising=False)

    expected_response = "Hello from Gemini"

    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=expected_response)

    mock_chat.return_value = fake_llm

    llm = base.GeminiLLM()

    response = await llm.ainvoke("Hello")

    assert response == expected_response
    fake_llm.ainvoke.assert_awaited_once_with("Hello")
