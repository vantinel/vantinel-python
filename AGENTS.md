# AGENTS.md — Vantinel Python SDK

## What Is This?

The official **Python SDK** for [Vantinel](https://vantinel.com) — a Real-Time AI Agent Observability & Guardrails Platform. Clients install this to send telemetry events to the Vantinel Collector for monitoring, anomaly detection, and policy enforcement.

## Project Structure

```
vantinel_sdk/
├── __init__.py           # Public API exports + convenience wrappers
├── config.py             # VantinelConfig dataclass + from_env() + builder methods
├── client.py             # VantinelClient: async HTTP client with circuit breaker
├── monitor.py            # VantinelMonitor: high-level API (watch_tool, decorators, wrap_openai)
├── types.py              # VantinelEvent, VantinelResponse, Decision enum, ModelPricing
├── cost.py               # Cost estimation from token counts + model pricing table
├── security.py           # HMAC signing, URL validation, nonce generation, API key redaction
├── errors.py             # Custom exceptions (ToolCallBlockedError, etc.)
└── integrations/
    ├── openai_agents.py  # OpenAI Agents SDK auto-instrumentation
    ├── anthropic.py      # Anthropic client wrapper
    ├── langgraph.py      # LangGraph integration
    ├── crewai.py         # CrewAI integration
    └── autogen.py        # AutoGen integration

tests/                    # pytest tests (asyncio_mode=auto)
examples/                 # Usage examples
```

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `VantinelConfig` | `config.py` | Dataclass with all SDK settings. Use `VantinelConfig.from_env()` for env-var-based init. Builder pattern: `.with_log()`, `.with_dry_run()`, etc. |
| `VantinelClient` | `client.py` | Async HTTP client (`httpx`). Sends events to collector. Has circuit breaker for fault tolerance. |
| `VantinelMonitor` | `monitor.py` | Main user-facing API. `watch_tool()` returns `ToolExecution` context. `wrap_openai()` / `wrap_langchain()` for auto-instrumentation. Supports singleton pattern. |
| `ToolExecution` | `monitor.py` | Returned by `watch_tool()`. Call `.success(result)` or `.error(msg)` to complete. |
| `VantinelEvent` | `types.py` | Event dataclass: `tool_name`, `tool_args_hash`, `latency_ms`, `estimated_cost`, `trace_payload` |

## Config (Environment Variables)

| Config | Env Var | Default |
|--------|---------|---------|
| `api_key` | `VANTINEL_API_KEY` | required |
| `project_id` | `VANTINEL_PROJECT_ID` | required |
| `collector_url` | `VANTINEL_COLLECTOR_URL` | `http://localhost:8000` |
| `agent_id` | `VANTINEL_AGENT_ID` | `default_agent` |
| `dry_run` | `VANTINEL_DRY_RUN` | `false` |
| `verbose` | `VANTINEL_VERBOSE` | `false` |
| `shadow_mode` | `VANTINEL_SHADOW_MODE` | `false` |
| `log` | `VANTINEL_LOG` | `false` — opt-in full request/response payload logging |

## Development Commands

```bash
# Install in dev mode
pip install -e .

# Run tests (uses pytest-asyncio with auto mode)
.venv/bin/python -m pytest tests/ -v

# Build distribution
python -m build
```

## Architecture Notes

- **Async-first**: `VantinelClient` uses `httpx.AsyncClient`. All event sending is async.
- **Sync support**: `wrap_openai()` handles both sync and async OpenAI clients. Sync paths use background threads for fire-and-forget telemetry.
- **Circuit breaker**: After 3 consecutive failures, SDK stops hitting collector for 30s. Fails open (allows all operations) during outage.
- **`trace` / `log`**: When either `trace=True` (on monitor) or `config.log=True`, the `trace_payload` field on events includes the full tool args and response (limited to 4KB for responses).
- **Security**: HMAC-SHA256 signing on every request. API key sent in header (redacted in error logs via `redact_api_key()`).
- **Cost estimation**: Built-in model pricing table in `types.py` / `cost.py` for OpenAI, Anthropic, Google models. Prefers provider-reported cost (e.g., OpenRouter `usage.cost`) over estimates.
