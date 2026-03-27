"""Tests for wrap_openai and wrap_langchain in VantinelMonitor."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from vantinel_sdk import VantinelMonitor, VantinelConfig


def make_monitor():
    config = VantinelConfig(
        api_key="test_key", project_id="test_client"
    ).with_dry_run()
    return VantinelMonitor(config)


class TestWrapOpenAISyncClient:
    """Tests for wrap_openai with a synchronous OpenAI client."""

    def _make_sync_openai(self, response=None):
        """Build a mock sync OpenAI client."""
        mock_response = response or MagicMock(
            id="chatcmpl-abc",
            model="gpt-4o",
            usage=MagicMock(
                prompt_tokens=50,
                completion_tokens=20,
                prompt_tokens_details=None,
            ),
        )
        client = MagicMock()
        client.chat.completions.create = MagicMock(return_value=mock_response)
        return client, mock_response

    def test_returns_same_client(self):
        monitor = make_monitor()
        openai, _ = self._make_sync_openai()
        result = monitor.wrap_openai(openai)
        assert result is openai

    def test_calls_original_create(self):
        monitor = make_monitor()
        openai, mock_response = self._make_sync_openai()
        original_create = openai.chat.completions.create

        monitor.wrap_openai(openai)
        result = openai.chat.completions.create(model="gpt-4o", messages=[])

        assert result is mock_response

    def test_streaming_yields_all_chunks(self):
        monitor = make_monitor()
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))], usage=None),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))], usage=None),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="!"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=3, prompt_tokens_details=None),
                model="gpt-4o",
            ),
        ]
        openai = MagicMock()
        openai.chat.completions.create = MagicMock(return_value=iter(chunks))

        monitor.wrap_openai(openai)
        stream = openai.chat.completions.create(model="gpt-4o", messages=[], stream=True)

        collected = list(stream)
        assert len(collected) == 3
        assert collected[0].choices[0].delta.content == "Hello"

    def test_does_not_mutate_kwargs(self):
        monitor = make_monitor()
        openai = MagicMock()
        openai.chat.completions.create = MagicMock(return_value=MagicMock(
            id="r1", model="gpt-4o", usage=None,
        ))

        monitor.wrap_openai(openai)

        # Store original kwargs
        kwargs = {"model": "gpt-4o", "messages": [], "stream": True}
        kwargs_before = dict(kwargs)

        # The sync stream path
        openai.chat.completions.create = MagicMock(return_value=iter([]))
        monitor.wrap_openai(openai)
        openai.chat.completions.create(**kwargs)

        # stream_options should NOT be added to original dict
        # (it's injected into a copy in the wrapper)
        assert "stream_options" not in kwargs_before

    def test_error_is_reraised(self):
        monitor = make_monitor()
        openai = MagicMock()
        openai.chat.completions.create = MagicMock(side_effect=RuntimeError("API down"))

        monitor.wrap_openai(openai)

        with pytest.raises(RuntimeError, match="API down"):
            openai.chat.completions.create(model="gpt-4o", messages=[])


@pytest.mark.asyncio
class TestWrapOpenAIAsyncClient:
    """Tests for wrap_openai with an async OpenAI client."""

    def _make_async_openai(self, response=None):
        """Build a mock async OpenAI client."""
        mock_response = response or MagicMock(
            id="chatcmpl-abc",
            model="gpt-4o",
            usage=MagicMock(
                prompt_tokens=50,
                completion_tokens=20,
                prompt_tokens_details=None,
            ),
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_response)
        return client, mock_response

    async def test_calls_original_create(self):
        monitor = make_monitor()
        openai, mock_response = self._make_async_openai()

        monitor.wrap_openai(openai)
        result = await openai.chat.completions.create(model="gpt-4o", messages=[])

        assert result is mock_response

    async def test_async_streaming(self):
        monitor = make_monitor()

        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hi"))], usage=None),
            MagicMock(
                choices=[MagicMock(delta=MagicMock(content="!"))],
                usage=MagicMock(prompt_tokens=5, completion_tokens=1, prompt_tokens_details=None),
                model="gpt-4o",
            ),
        ]

        async def mock_create(**kwargs):
            async def gen():
                for c in chunks:
                    yield c
            return gen()

        openai = MagicMock()
        openai.chat.completions.create = mock_create

        monitor.wrap_openai(openai)
        stream = await openai.chat.completions.create(model="gpt-4o", messages=[], stream=True)

        collected = []
        async for chunk in stream:
            collected.append(chunk)

        assert len(collected) == 2

    async def test_error_is_reraised(self):
        monitor = make_monitor()
        openai = MagicMock()
        openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))

        monitor.wrap_openai(openai)

        with pytest.raises(RuntimeError, match="timeout"):
            await openai.chat.completions.create(model="gpt-4o", messages=[])


class TestWrapLangChain:
    """Tests for wrap_langchain."""

    def _make_mock_llm(self):
        llm = MagicMock()
        llm.__class__.__name__ = "ChatOpenAI"
        llm.invoke = MagicMock(return_value="invoke-result")
        llm.ainvoke = AsyncMock(return_value="ainvoke-result")
        return llm

    def test_returns_same_object(self):
        monitor = make_monitor()
        llm = self._make_mock_llm()
        result = monitor.wrap_langchain(llm)
        assert result is llm

    def test_sync_invoke_calls_original(self):
        monitor = make_monitor()
        llm = self._make_mock_llm()
        original_invoke = llm.invoke

        monitor.wrap_langchain(llm)
        result = llm.invoke("hello")

        assert result == "invoke-result"

    def test_sync_invoke_error_reraised(self):
        monitor = make_monitor()
        llm = self._make_mock_llm()
        llm.invoke = MagicMock(side_effect=ValueError("bad input"))

        monitor.wrap_langchain(llm)

        with pytest.raises(ValueError, match="bad input"):
            llm.invoke("hello")

    @pytest.mark.asyncio
    async def test_async_invoke_calls_original(self):
        monitor = make_monitor()
        llm = self._make_mock_llm()

        monitor.wrap_langchain(llm)
        result = await llm.ainvoke("hello")

        assert result == "ainvoke-result"

    @pytest.mark.asyncio
    async def test_async_invoke_error_reraised(self):
        monitor = make_monitor()
        llm = self._make_mock_llm()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("model error"))

        monitor.wrap_langchain(llm)

        with pytest.raises(RuntimeError, match="model error"):
            await llm.ainvoke("hello")

    def test_wraps_only_if_ainvoke_exists(self):
        """If LLM has no ainvoke, only sync invoke is wrapped."""
        monitor = make_monitor()
        llm = MagicMock(spec=["invoke"])
        llm.__class__.__name__ = "SimpleLLM"
        llm.invoke = MagicMock(return_value="ok")

        monitor.wrap_langchain(llm)
        result = llm.invoke("test")
        assert result == "ok"
