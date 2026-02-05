"""Tool call parser for LLM responses."""

import json
import re

# Tool-related tokens
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


def parse_tool_calls(response: str) -> list[dict]:
    """Parse tool calls from assistant response.

    Supports both formats:
    1. Pythonic: [web_search(queries=["query1", "query2"])]
    2. JSON: {"name": "web_search", "arguments": {"queries": ["query1"]}}
    """
    tool_calls = []

    # Find content between tool call markers
    pattern = re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END)
    matches = re.findall(pattern, response, re.DOTALL)

    for match in matches:
        match = match.strip()

        # Try JSON format first
        if match.startswith("{") or match.startswith("["):
            parsed = _parse_json_tool_calls(match)
            if parsed:
                tool_calls.extend(parsed)
                continue

        # Fall back to Pythonic format
        parsed = _parse_pythonic_tool_calls(match)
        tool_calls.extend(parsed)

    return tool_calls


def _parse_json_tool_calls(content: str) -> list[dict]:
    """Parse JSON format tool calls."""
    tool_calls = []

    try:
        data = json.loads(content)
        # Handle single object or array
        if isinstance(data, dict):
            data = [data]

        for item in data:
            if isinstance(item, dict) and "name" in item:
                tool_calls.append({
                    "name": item["name"],
                    "arguments": item.get("arguments", {}),
                })
    except json.JSONDecodeError:
        pass

    return tool_calls


def _parse_pythonic_tool_calls(content: str) -> list[dict]:
    """Parse Pythonic format tool calls."""
    tool_calls = []

    # Parse individual function calls
    # Pattern: function_name(param="value", ...)
    func_pattern = r"(\w+)\((.*?)\)"
    func_matches = re.findall(func_pattern, content, re.DOTALL)

    for func_name, params_str in func_matches:
        tool_call = {"name": func_name, "arguments": {}}

        if params_str.strip():
            # Parse named parameters
            # Handle: queries=["q1", "q2"] or thought="..."
            param_pattern = r"(\w+)=(.+?)(?=,\s*\w+=|$)"
            param_matches = re.findall(param_pattern, params_str, re.DOTALL)

            for param_name, param_value in param_matches:
                param_value = param_value.strip()
                # Try to parse as JSON
                try:
                    tool_call["arguments"][param_name] = json.loads(param_value)
                except json.JSONDecodeError:
                    # If not JSON, try as string (remove quotes)
                    if param_value.startswith('"') and param_value.endswith('"'):
                        tool_call["arguments"][param_name] = param_value[1:-1]
                    else:
                        tool_call["arguments"][param_name] = param_value

        tool_calls.append(tool_call)

    return tool_calls
