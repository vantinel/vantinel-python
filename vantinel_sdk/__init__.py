"""
Vantinel SDK for Python

Lightweight observability and guardrails SDK for AI agents.
"""

__version__ = "0.5.3"

from .monitor import VantinelMonitor, ToolExecution
from .config import VantinelConfig
from .types import Decision, VantinelEvent, VantinelResponse
from .errors import (
    VantinelError,
    ToolCallBlockedError,
    CollectorUnavailableError,
    ConfigurationError,
)
from .security import hmac_sign, validate_collector_url, redact_api_key

# New in 0.5.3: Framework integrations
from .integrations import (
    patch_openai_agents,
    VantinelTracingProcessor,
    wrap_anthropic,
    VantinelCallbackHandler,
    wrap_langgraph,
    VantinelHook,
)

__all__ = [
    "VantinelMonitor",
    "ToolExecution",
    "VantinelConfig",
    "Decision",
    "VantinelEvent",
    "VantinelResponse",
    "VantinelError",
    "ToolCallBlockedError",
    "CollectorUnavailableError",
    "ConfigurationError",
    "hmac_sign",
    "validate_collector_url",
    "redact_api_key",
    # New in 0.5.3: Framework integrations
    "patch_openai_agents",
    "VantinelTracingProcessor",
    "wrap_anthropic",
    "VantinelCallbackHandler",
    "wrap_langgraph",
    "VantinelHook",
    # New in 0.2.0-beta
    "wrap_openai",
    "wrap_langchain",
    "ping",
    "capture_error",
    "__version__",
]


def wrap_openai(monitor: VantinelMonitor, openai_client):
    """Module-level convenience wrapper for monitor.wrap_openai()."""
    return monitor.wrap_openai(openai_client)


def wrap_langchain(monitor: VantinelMonitor, llm):
    """Module-level convenience wrapper for monitor.wrap_langchain()."""
    return monitor.wrap_langchain(llm)


async def ping(monitor: VantinelMonitor) -> dict:
    """Module-level convenience wrapper for monitor.ping()."""
    return await monitor.ping()


async def capture_error(monitor: VantinelMonitor, tool_name: str, error: Exception, metadata=None) -> None:
    """Module-level convenience wrapper for monitor.capture_error()."""
    return await monitor.capture_error(tool_name, error, metadata)
