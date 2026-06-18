# 🎭 Random Persona MCP Server

MCP (Model Context Protocol) server exposing the six-layer persona engine.

## Quick Start

```bash
pip install fastmcp
python server.py
# → MCP server on http://127.0.0.1:4568
```

## Tools

| Tool | Description |
|------|------------|
| `persona_inject` | Get system-prompt injection for current session state |
| `persona_post_process` | Post-process LLM response (silence, filler) |
| `persona_status` | Get human-readable state display |
| `persona_command` | Handle /persona commands |

## Usage with AstrBot

Configure `astrbot_plugin_random_persona` to point at this server:

```json
{
  "mcp_url": "http://127.0.0.1:4568"
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PERSONA_PORT` | `4568` | Server port |
| `PERSONA_HOST` | `127.0.0.1` | Bind address |
| `PERSONA_DATA_DIR` | `./data` | State persistence directory |
| `MCP_TRANSPORT` | `sse` | Transport: `sse` or `stdio` |

## Architecture

```
AstrBot plugin  ──MCP──►  persona_inject(session_id, user_id, message)
                         persona_post_process(session_id, response)
                         persona_command(...)

Internal pipeline:
  Mood drift → Appraisal → Emotion decay → Relationship → SpeechAct → Lexicon → Prompt
```

## License

MIT
