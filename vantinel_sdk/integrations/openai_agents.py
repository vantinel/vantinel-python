"""OpenAI Agents SDK integration for Vantinel.

Usage:
    import vantinel_sdk
    from vantinel_sdk.integrations import patch_openai_agents

    monitor = vantinel_sdk.VantinelMonitor(vantinel_sdk.VantinelConfig.from_env())
    patch_openai_agents(monitor)

    # Now all openai-agents traces are automatically monitored
    from agents import Agent, Runner
    agent = Agent(name="my-agent", instructions="...")
    result = await Runner.run(agent, "Hello")
"""

import time
import json
import asyncio
import threading
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from ..monitor import VantinelMonitor


class VantinelTracingProcessor:
    """TracingProcessor that forwards OpenAI Agents SDK spans to Vantinel.

    Implements the openai-agents TracingProcessor interface:
    - on_trace_start(trace)
    - on_trace_end(trace)
    - on_span_start(span)
    - on_span_end(span)
    """

    def __init__(self, monitor: "VantinelMonitor"):
        self.monitor = monitor
        self._span_starts: dict = {}
        self._lock = threading.Lock()

    def on_trace_start(self, trace: Any) -> None:
        """Called when a new trace (agent run) starts."""
        pass

    def on_trace_end(self, trace: Any) -> None:
        """Called when a trace (agent run) completes."""
        pass

    def on_span_start(self, span: Any) -> None:
        """Called when a span starts — record start time."""
        span_id = getattr(span, "span_id", None) or id(span)
        with self._lock:
            self._span_starts[span_id] = time.time()

    def on_span_end(self, span: Any) -> None:
        """Called when a span ends — send telemetry to Vantinel."""
        span_id = getattr(span, "span_id", None) or id(span)
        with self._lock:
            start = self._span_starts.pop(span_id, None)

        latency_ms = (time.time() - start) * 1000 if start else None

        # Extract span details
        span_data = getattr(span, "span_data", None)
        span_type = type(span_data).__name__ if span_data else "unknown_span"

        # Map span types to tool names
        tool_name = self._extract_tool_name(span, span_data, span_type)
        if tool_name is None:
            return  # Skip spans we don't care about

        metadata = self._extract_metadata(span, span_data, span_type)
        if latency_ms is not None:
            metadata["latency_ms"] = latency_ms

        # Estimate cost from token usage if available
        estimated_cost = self._extract_cost(span_data)

        self._send_telemetry(tool_name, json.dumps({}), estimated_cost, metadata, latency_ms)

    def _extract_tool_name(self, span: Any, span_data: Any, span_type: str) -> Optional[str]:
        """Map OpenAI Agents span types to Vantinel tool names."""
        if span_type == "AgentSpanData":
            agent_name = getattr(span_data, "name", None) or "agent"
            return f"agent_run_{agent_name}"
        elif span_type == "FunctionSpanData":
            fn_name = getattr(span_data, "name", None) or "function_call"
            return f"tool_call_{fn_name}"
        elif span_type == "GenerationSpanData":
            model = getattr(span_data, "model", None) or "unknown"
            return f"llm_generation_{model}"
        elif span_type == "HandoffSpanData":
            target = getattr(span_data, "to_agent", None) or "unknown"
            return f"handoff_to_{target}"
        elif span_type == "GuardrailSpanData":
            guardrail_name = getattr(span_data, "name", None) or "guardrail"
            return f"guardrail_{guardrail_name}"
        # Skip internal/trace-level spans
        return None

    def _extract_metadata(self, span: Any, span_data: Any, span_type: str) -> dict:
        """Extract relevant metadata from span."""
        meta: dict = {"span_type": span_type, "framework": "openai-agents"}
        if span_data:
            for attr in ("name", "model", "to_agent", "from_agent", "triggered_by"):
                val = getattr(span_data, attr, None)
                if val is not None:
                    meta[attr] = val
        return meta

    def _extract_cost(self, span_data: Any) -> Optional[float]:
        """Extract estimated cost from generation span usage."""
        if span_data is None:
            return None
        usage = getattr(span_data, "usage", None)
        if usage is None:
            return None
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        model = getattr(span_data, "model", "gpt-4o") or "gpt-4o"
        try:
            from ..cost import estimate_cost
            return estimate_cost(model, input_tokens, output_tokens)
        except Exception:
            return None

    def _send_telemetry(
        self,
        tool_name: str,
        args_str: str,
        estimated_cost: Optional[float],
        metadata: dict,
        latency_ms: Optional[float],
    ) -> None:
        """Send telemetry fire-and-forget."""
        monitor = self.monitor

        async def _send():
            try:
                execution = await monitor.watch_tool(
                    tool_name=tool_name,
                    tool_args=args_str,
                    estimated_cost=estimated_cost,
                    metadata=metadata,
                )
                await execution.success(metadata={"latency_ms": latency_ms})
            except Exception:
                pass

        def _run_in_thread():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_send())
            finally:
                loop.close()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send())
        except RuntimeError:
            t = threading.Thread(target=_run_in_thread, daemon=True)
            t.start()


def patch_openai_agents(monitor: "VantinelMonitor") -> "VantinelTracingProcessor":
    """Register a VantinelTracingProcessor with the OpenAI Agents SDK.

    Args:
        monitor: VantinelMonitor instance to send telemetry through

    Returns:
        The registered VantinelTracingProcessor instance

    Raises:
        ImportError: If the openai-agents package is not installed

    Example:
        monitor = VantinelMonitor(VantinelConfig.from_env())
        patch_openai_agents(monitor)
        # Now run your agents normally — telemetry flows automatically
    """
    try:
        from agents.tracing import add_trace_processor
    except ImportError:
        raise ImportError(
            "openai-agents package is required. Install with: pip install openai-agents"
        )

    processor = VantinelTracingProcessor(monitor)
    add_trace_processor(processor)
    return processor
