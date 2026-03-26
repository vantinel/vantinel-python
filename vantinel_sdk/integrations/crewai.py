"""CrewAI integration for Vantinel.

Usage:
    from crewai import Crew, Agent, Task
    import vantinel_sdk
    from vantinel_sdk.integrations import VantinelCallbackHandler

    monitor = vantinel_sdk.VantinelMonitor(vantinel_sdk.VantinelConfig.from_env())
    handler = VantinelCallbackHandler(monitor)

    crew = Crew(
        agents=[...],
        tasks=[...],
        step_callback=handler.on_step,
        task_callback=handler.on_task_complete,
    )
    result = crew.kickoff()
"""

import time
import json
import asyncio
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..monitor import VantinelMonitor


class VantinelCallbackHandler:
    """CrewAI callback handler that forwards agent/task events to Vantinel.

    Hooks into CrewAI's step_callback and task_callback interfaces.
    """

    def __init__(self, monitor: "VantinelMonitor"):
        self.monitor = monitor
        self._step_starts: dict = {}
        self._lock = threading.Lock()

    def on_step(self, step_output: Any) -> None:
        """Called by CrewAI's step_callback after each agent step.

        Args:
            step_output: AgentFinish or AgentAction from langchain agents
        """
        tool_name = "crewai_agent_step"
        metadata: dict = {"framework": "crewai"}

        # Extract tool name from action if available
        action_tool = getattr(step_output, "tool", None)
        if action_tool:
            tool_name = f"crewai_tool_{action_tool}"
            tool_input = getattr(step_output, "tool_input", {})
            metadata["tool_input_keys"] = list(tool_input.keys()) if isinstance(tool_input, dict) else []

        # Extract return values from finish
        output = getattr(step_output, "return_values", None) or getattr(step_output, "log", None)
        if output:
            metadata["has_output"] = True

        self._send_telemetry(tool_name, json.dumps({}), None, metadata)

    def on_task_complete(self, task_output: Any) -> None:
        """Called by CrewAI's task_callback when a task finishes.

        Args:
            task_output: TaskOutput from crewai
        """
        description = getattr(task_output, "description", None) or "unknown_task"
        # Sanitize task name for use as tool_name
        safe_name = description[:40].replace(" ", "_").replace("/", "_")
        tool_name = f"crewai_task_{safe_name}"

        metadata: dict = {
            "framework": "crewai",
            "task_description": description[:200] if description else None,
        }
        agent_name = getattr(task_output, "agent", None)
        if agent_name:
            metadata["agent"] = str(agent_name)[:100]

        self._send_telemetry(tool_name, json.dumps({}), None, metadata)

    def _send_telemetry(
        self,
        tool_name: str,
        args_str: str,
        estimated_cost: Optional[float],
        metadata: dict,
    ) -> None:
        monitor = self.monitor

        async def _send():
            try:
                execution = await monitor.watch_tool(
                    tool_name=tool_name,
                    tool_args=args_str,
                    estimated_cost=estimated_cost,
                    metadata=metadata,
                )
                await execution.success()
            except Exception:
                pass

        def _run():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_send())
            finally:
                loop.close()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send())
        except RuntimeError:
            t = threading.Thread(target=_run, daemon=True)
            t.start()
