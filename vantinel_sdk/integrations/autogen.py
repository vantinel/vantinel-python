"""AutoGen integration for Vantinel.

Usage:
    import autogen
    import vantinel_sdk
    from vantinel_sdk.integrations import VantinelHook

    monitor = vantinel_sdk.VantinelMonitor(vantinel_sdk.VantinelConfig.from_env())
    hook = VantinelHook(monitor)

    agent = autogen.AssistantAgent(name="assistant", llm_config={...})
    agent.register_hook("process_message_before_send", hook.on_message_before_send)
    agent.register_hook("process_all_messages_before_reply", hook.on_before_reply)
"""

import time
import json
import asyncio
import threading
from typing import TYPE_CHECKING, Any, Optional, List

if TYPE_CHECKING:
    from ..monitor import VantinelMonitor


class VantinelHook:
    """AutoGen hook that forwards agent message events to Vantinel.

    Register with agent.register_hook() for automatic monitoring.

    Supported hook points:
    - process_message_before_send: Tracks outgoing messages
    - process_all_messages_before_reply: Tracks incoming message context
    """

    def __init__(self, monitor: "VantinelMonitor"):
        self.monitor = monitor

    def on_message_before_send(self, message: Any, recipient: Any, silent: bool) -> Any:
        """Hook for process_message_before_send.

        Called before an agent sends a message. Returns the message unchanged.
        """
        agent_name = getattr(recipient, "name", None) or "unknown_agent"
        tool_name = f"autogen_send_to_{agent_name}"

        metadata: dict = {
            "framework": "autogen",
            "recipient": agent_name,
            "silent": silent,
        }
        if isinstance(message, dict):
            content = message.get("content", "")
            metadata["content_length"] = len(str(content)) if content else 0
            if message.get("tool_calls"):
                metadata["has_tool_calls"] = True
                metadata["tool_call_count"] = len(message["tool_calls"])

        self._send_telemetry(tool_name, json.dumps({}), None, metadata)
        return message  # Always return message unchanged

    def on_before_reply(self, messages: List[Any], sender: Any, config: Any) -> tuple:
        """Hook for process_all_messages_before_reply.

        Called before an agent generates a reply. Returns (False, None) to not override reply.
        """
        sender_name = getattr(sender, "name", None) or "unknown_sender"
        tool_name = f"autogen_receive_from_{sender_name}"

        metadata: dict = {
            "framework": "autogen",
            "sender": sender_name,
            "message_count": len(messages) if messages else 0,
        }

        self._send_telemetry(tool_name, json.dumps({}), None, metadata)
        return False, None  # Don't override the reply

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
