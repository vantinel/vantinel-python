"""Vantinel integrations for major AI agent frameworks."""

from .openai_agents import patch_openai_agents, VantinelTracingProcessor
from .anthropic import wrap_anthropic
from .crewai import VantinelCallbackHandler
from .langgraph import wrap_langgraph
from .autogen import VantinelHook

__all__ = [
    "patch_openai_agents",
    "VantinelTracingProcessor",
    "wrap_anthropic",
    "VantinelCallbackHandler",
    "wrap_langgraph",
    "VantinelHook",
]
