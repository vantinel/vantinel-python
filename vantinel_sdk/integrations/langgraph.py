"""LangGraph integration for Vantinel.

Usage:
    from langgraph.graph import StateGraph
    import vantinel_sdk
    from vantinel_sdk.integrations import wrap_langgraph

    monitor = vantinel_sdk.VantinelMonitor(vantinel_sdk.VantinelConfig.from_env())

    builder = StateGraph(MyState)
    builder.add_node("agent", agent_node)
    graph = builder.compile()

    # Wrap the compiled graph
    graph = wrap_langgraph(monitor, graph)
    result = graph.invoke({"messages": [...]})
"""

import time
import json
import asyncio
import functools
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..monitor import VantinelMonitor


def wrap_langgraph(monitor: "VantinelMonitor", graph: Any) -> Any:
    """Wrap a compiled LangGraph to auto-monitor node transitions and invocations.

    Patches graph.invoke() and graph.ainvoke() to track:
    - Total invocation time and cost
    - Node transitions (via stream events if available)

    Args:
        monitor: VantinelMonitor instance
        graph: A compiled LangGraph (result of StateGraph.compile())

    Returns:
        The patched graph (same object, modified in-place)
    """
    original_invoke = getattr(graph, "invoke", None)
    original_ainvoke = getattr(graph, "ainvoke", None)
    original_stream = getattr(graph, "stream", None)
    original_astream = getattr(graph, "astream", None)

    graph_name = getattr(graph, "name", None) or type(graph).__name__

    if original_invoke:
        @functools.wraps(original_invoke)
        def monitored_invoke(input_data, config=None, **kwargs):
            start = time.time()
            try:
                result = original_invoke(input_data, config, **kwargs)
                latency_ms = (time.time() - start) * 1000
                monitor._fire_and_forget_sync(
                    f"langgraph_invoke_{graph_name}", json.dumps({}),
                    graph_name, latency_ms, None
                )
                return result
            except Exception as e:
                monitor._fire_and_forget_sync(
                    f"langgraph_invoke_{graph_name}", json.dumps({}),
                    graph_name, (time.time() - start) * 1000, None, error=str(e)
                )
                raise

        graph.invoke = monitored_invoke

    if original_ainvoke:
        @functools.wraps(original_ainvoke)
        async def monitored_ainvoke(input_data, config=None, **kwargs):
            execution = await monitor.watch_tool(
                tool_name=f"langgraph_invoke_{graph_name}",
                tool_args=json.dumps({}),
                metadata={"framework": "langgraph", "graph": graph_name},
            )
            try:
                result = await original_ainvoke(input_data, config, **kwargs)
                await execution.success(result)
                return result
            except Exception as e:
                await execution.error(str(e))
                raise

        graph.ainvoke = monitored_ainvoke

    if original_stream:
        @functools.wraps(original_stream)
        def monitored_stream(input_data, config=None, **kwargs):
            node_count = 0
            start = time.time()
            try:
                for chunk in original_stream(input_data, config, **kwargs):
                    node_count += 1
                    # Each chunk is a dict {node_name: state_update}
                    for node_name in chunk:
                        monitor._fire_and_forget_sync(
                            f"langgraph_node_{node_name}", json.dumps({}),
                            node_name, 0.0, None
                        )
                    yield chunk
            finally:
                latency_ms = (time.time() - start) * 1000
                monitor._fire_and_forget_sync(
                    f"langgraph_stream_{graph_name}", json.dumps({"nodes_executed": node_count}),
                    graph_name, latency_ms, None
                )

        graph.stream = monitored_stream

    if original_astream:
        @functools.wraps(original_astream)
        async def monitored_astream(input_data, config=None, **kwargs):
            node_count = 0
            start = time.time()
            try:
                async for chunk in original_astream(input_data, config, **kwargs):
                    node_count += 1
                    for node_name in chunk:
                        # Fire-and-forget node telemetry
                        monitor._schedule_task(
                            _send_node_event(monitor, node_name)
                        )
                    yield chunk
            finally:
                latency_ms = (time.time() - start) * 1000
                monitor._schedule_task(
                    _send_stream_complete(monitor, graph_name, node_count, latency_ms)
                )

        graph.astream = monitored_astream

    return graph


async def _send_node_event(monitor: "VantinelMonitor", node_name: str) -> None:
    try:
        execution = await monitor.watch_tool(
            tool_name=f"langgraph_node_{node_name}",
            tool_args=json.dumps({}),
            metadata={"framework": "langgraph", "node": node_name},
        )
        await execution.success()
    except Exception:
        pass


async def _send_stream_complete(monitor: "VantinelMonitor", graph_name: str, node_count: int, latency_ms: float) -> None:
    try:
        execution = await monitor.watch_tool(
            tool_name=f"langgraph_stream_{graph_name}",
            tool_args=json.dumps({"nodes_executed": node_count}),
            metadata={"framework": "langgraph", "graph": graph_name, "nodes_executed": node_count},
        )
        await execution.success(metadata={"latency_ms": latency_ms})
    except Exception:
        pass
