"""Anthropic SDK integration for Vantinel.

Usage:
    import anthropic
    import vantinel_sdk
    from vantinel_sdk.integrations import wrap_anthropic

    monitor = vantinel_sdk.VantinelMonitor(vantinel_sdk.VantinelConfig.from_env())
    client = wrap_anthropic(monitor, anthropic.Anthropic())

    # Now all messages.create() calls are automatically monitored
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

import time
import json
import asyncio
import threading
import functools
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..monitor import VantinelMonitor


def wrap_anthropic(monitor: "VantinelMonitor", anthropic_client):
    """Wrap an Anthropic client to auto-monitor all messages.create() calls.

    Supports both sync (anthropic.Anthropic) and async (anthropic.AsyncAnthropic) clients.
    Monitors streaming and non-streaming responses.

    Args:
        monitor: VantinelMonitor instance
        anthropic_client: anthropic.Anthropic() or anthropic.AsyncAnthropic() instance

    Returns:
        The patched client (same object, modified in-place)
    """
    import inspect

    original_create = anthropic_client.messages.create
    is_async = inspect.iscoroutinefunction(original_create) or inspect.iscoroutinefunction(
        getattr(original_create, "__wrapped__", None)
    )

    def _extract_cost(model: str, usage) -> Optional[float]:
        if usage is None:
            return None
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        try:
            from ..cost import estimate_cost
            return estimate_cost(
                model_name=model,
                input_tokens=input_tokens + cache_creation,
                output_tokens=output_tokens,
                cached_tokens=cache_read,
            )
        except Exception:
            return None

    def _build_args_str(model: str, messages: list) -> str:
        import hashlib
        msg_hash = hashlib.md5(str(messages).encode()).hexdigest()[:8]
        return json.dumps({"model": model, "messages_count": len(messages), "msg_hash": msg_hash})

    def _extract_tool_uses(response) -> list:
        """Extract tool_use content blocks from response."""
        tool_uses = []
        content = getattr(response, "content", []) or []
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                tool_uses.append(getattr(block, "name", "unknown_tool"))
        return tool_uses

    if is_async:
        @functools.wraps(original_create)
        async def async_monitored_create(**kwargs):
            model = kwargs.get("model", "claude-opus-4-6")
            messages = kwargs.get("messages", [])
            tool_name = f"anthropic_messages_{model}"
            args_str = _build_args_str(model, messages)
            is_stream = kwargs.get("stream", False)

            execution = await monitor.watch_tool(
                tool_name=tool_name,
                tool_args=args_str,
                metadata={"model": model, "messages_count": len(messages), "framework": "anthropic"},
            )
            start = time.time()
            try:
                result = await original_create(**kwargs)
                latency_ms = (time.time() - start) * 1000

                if is_stream:
                    async def _stream_wrapper():
                        usage = None
                        try:
                            async for event in result:
                                if hasattr(event, "usage"):
                                    usage = event.usage
                                yield event
                        finally:
                            cost = _extract_cost(model, usage)
                            monitor._schedule_task(
                                execution.success(metadata={"latency_ms": (time.time() - start) * 1000, "cost_usd": cost})
                            )
                    return _stream_wrapper()

                cost = _extract_cost(model, getattr(result, "usage", None))
                tool_uses = _extract_tool_uses(result)
                meta = {"latency_ms": latency_ms, "cost_usd": cost}
                if tool_uses:
                    meta["tool_uses"] = tool_uses
                stop_reason = getattr(result, "stop_reason", None)
                if stop_reason:
                    meta["stop_reason"] = stop_reason
                await execution.success(result, metadata=meta)
                return result
            except Exception as e:
                await execution.error(str(e))
                raise

        anthropic_client.messages.create = async_monitored_create
    else:
        @functools.wraps(original_create)
        def sync_monitored_create(**kwargs):
            model = kwargs.get("model", "claude-opus-4-6")
            messages = kwargs.get("messages", [])
            tool_name = f"anthropic_messages_{model}"
            args_str = _build_args_str(model, messages)
            is_stream = kwargs.get("stream", False)
            start = time.time()
            try:
                result = original_create(**kwargs)
                latency_ms = (time.time() - start) * 1000

                if is_stream:
                    def _stream_wrapper():
                        usage = None
                        try:
                            for event in result:
                                if hasattr(event, "usage"):
                                    usage = event.usage
                                yield event
                        finally:
                            cost = _extract_cost(model, usage)
                            monitor._fire_and_forget_sync(
                                tool_name, args_str, model,
                                (time.time() - start) * 1000, cost
                            )
                    return _stream_wrapper()

                cost = _extract_cost(model, getattr(result, "usage", None))
                monitor._fire_and_forget_sync(tool_name, args_str, model, latency_ms, cost)
                return result
            except Exception as e:
                monitor._fire_and_forget_sync(
                    tool_name, args_str, model,
                    (time.time() - start) * 1000, None, error=str(e)
                )
                raise

        anthropic_client.messages.create = sync_monitored_create

    return anthropic_client
