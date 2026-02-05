"""Tool definitions for the search agent."""

from typing import Literal


def get_tools_definition(
    mode: Literal["speed", "balanced", "quality"] = "balanced",
) -> list[dict]:
    """Get tool definitions for the search agent."""
    tools = [
        {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search queries (max 10)",
                    }
                },
                "required": ["queries"],
            },
        },
        {
            "name": "done",
            "description": "Signal that research is complete",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    if mode in ("balanced", "quality"):
        tools.insert(
            0,
            {
                "name": "__reasoning_preamble",
                "description": "Express your reasoning before each tool call",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your reasoning for the next step",
                        }
                    },
                    "required": ["thought"],
                },
            },
        )

    return tools
