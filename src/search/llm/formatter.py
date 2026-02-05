"""Prompt formatter with special tokens."""

import json
import re
from datetime import datetime
from typing import Literal


class PromptFormatter:
    """Formats prompts for LLM chat template.

    Note: Chat template tokens (<|im_start|>, <|im_end|>, etc.) are NOT included
    in the output. The llama-server applies these automatically from the GGUF
    model's embedded chat template.
    """

    # Tool-related tokens (used in prompt content, not for message framing)
    TOOL_CALL_START = "<|tool_call_start|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_LIST_START = "<|tool_list_start|>"
    TOOL_LIST_END = "<|tool_list_end|>"

    def __init__(
        self, mode: Literal["speed", "balanced", "quality"] = "balanced"
    ) -> None:
        self.mode = mode

    def get_tools_definition(self) -> list[dict]:
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

        if self.mode in ("balanced", "quality"):
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

    def format_system_prompt(
        self,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Generate system prompt based on mode."""
        today = datetime.now().strftime("%B %d, %Y")
        tools = self.get_tools_definition()
        tool_desc = json.dumps(tools, indent=2)

        if self.mode == "speed":
            return self._get_speed_prompt(tool_desc, iteration, max_iterations, today)
        elif self.mode == "balanced":
            return self._get_balanced_prompt(
                tool_desc, iteration, max_iterations, today
            )
        else:
            return self._get_quality_prompt(tool_desc, iteration, max_iterations, today)


    def parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from assistant response.

        Supports both formats:
        1. Pythonic: [web_search(queries=["query1", "query2"])]
        2. JSON: {"name": "web_search", "arguments": {"queries": ["query1"]}}
        """
        tool_calls = []

        # Find content between tool call markers
        pattern = (
            re.escape(self.TOOL_CALL_START) + r"(.*?)" + re.escape(self.TOOL_CALL_END)
        )
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            match = match.strip()

            # Try JSON format first
            if match.startswith("{") or match.startswith("["):
                parsed = self._parse_json_tool_calls(match)
                if parsed:
                    tool_calls.extend(parsed)
                    continue

            # Fall back to Pythonic format
            parsed = self._parse_pythonic_tool_calls(match)
            tool_calls.extend(parsed)

        return tool_calls

    def _parse_json_tool_calls(self, content: str) -> list[dict]:
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

    def _parse_pythonic_tool_calls(self, content: str) -> list[dict]:
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

    def _get_speed_prompt(
        self, tool_desc: str, iteration: int, max_iterations: int, today: str
    ) -> str:
        """Generate speed mode system prompt."""
        return f"""You are an action orchestrator. Your job is to fulfill user requests by selecting and executing the available tools—no free-form replies.

Today's date: {today}

You are currently on iteration {iteration + 1} of your research process and have {max_iterations} total iterations so act efficiently.
When you are finished, you must call the `done` tool. Never output text directly.

<goal>
Fulfill the user's request as quickly as possible using the available tools.
Call tools to gather information or perform tasks as needed.
</goal>

<core_principle>
Your knowledge is outdated; use web search to ground answers even for seemingly basic facts.
</core_principle>

{self.TOOL_LIST_START}
{tool_desc}
{self.TOOL_LIST_END}

<response_protocol>
- NEVER output normal text to the user. ONLY call tools using {self.TOOL_CALL_START} and {self.TOOL_CALL_END} tokens.
- Default to web_search when information is missing or stale; keep queries targeted (max 10 per call).
- Call done when you have gathered enough to answer or performed the required actions.
</response_protocol>"""

    def _get_balanced_prompt(
        self, tool_desc: str, iteration: int, max_iterations: int, today: str
    ) -> str:
        """Generate balanced mode system prompt."""
        return f"""You are an action orchestrator. Your job is to fulfill user requests by reasoning briefly and executing the available tools—no free-form replies.

Today's date: {today}

You are currently on iteration {iteration + 1} of your research process and have {max_iterations} total iterations so act efficiently.
When you are finished, you must call the `done` tool. Never output text directly.

<goal>
Fulfill the user's request with concise reasoning plus focused actions.
You must call the __reasoning_preamble tool before every tool call in this assistant turn.
Alternate: __reasoning_preamble → tool → __reasoning_preamble → tool ... and finish with __reasoning_preamble → done.
</goal>

<core_principle>
Your knowledge is outdated; use web search to ground answers even for seemingly basic facts.
You can call at most 6 tools total per turn.
</core_principle>

{self.TOOL_LIST_START}
YOU MUST CALL __reasoning_preamble BEFORE EVERY TOOL CALL IN THIS ASSISTANT TURN.
{tool_desc}
{self.TOOL_LIST_END}

<response_protocol>
- NEVER output normal text to the user. ONLY call tools using {self.TOOL_CALL_START} and {self.TOOL_CALL_END} tokens.
- Start with __reasoning_preamble and call it before every tool call (including done).
- Default to web_search when information is missing or stale; keep queries targeted (max 10 per call).
- Call done only after you have the needed info or actions completed.
</response_protocol>"""

    def _get_quality_prompt(
        self, tool_desc: str, iteration: int, max_iterations: int, today: str
    ) -> str:
        """Generate quality mode system prompt."""
        return f"""You are a deep-research orchestrator. Your job is to fulfill user requests with thorough, comprehensive research—no free-form replies.

Today's date: {today}

You are currently on iteration {iteration + 1} of your research process and have {max_iterations} total iterations.
When you are finished, you must call the `done` tool. Never output text directly.

<goal>
Conduct the deepest, most thorough research possible. Leave no stone unturned.
Follow an iterative reason-act loop: call __reasoning_preamble before every tool call.
Finish with done only when you have comprehensive, multi-angle information.
</goal>

<core_principle>
Your knowledge is outdated; always use the available tools to ground answers.
This is DEEP RESEARCH mode—be exhaustive. Explore multiple angles: definitions, features, comparisons, recent news, expert opinions, use cases, limitations.
You can call up to 10 tools total per turn.
</core_principle>

{self.TOOL_LIST_START}
YOU MUST CALL __reasoning_preamble BEFORE EVERY TOOL CALL IN THIS ASSISTANT TURN.
{tool_desc}
{self.TOOL_LIST_END}

<research_strategy>
For any topic, consider searching:
1. Core definition/overview - What is it?
2. Features/capabilities - What can it do?
3. Comparisons - How does it compare to alternatives?
4. Recent news/updates - What's the latest?
5. Reviews/opinions - What do experts say?
</research_strategy>

<response_protocol>
- NEVER output normal text to the user. ONLY call tools using {self.TOOL_CALL_START} and {self.TOOL_CALL_END} tokens.
- Follow an iterative loop: __reasoning_preamble → tool call → __reasoning_preamble → tool call → ... → done.
- Aim for 4-7 information-gathering calls covering different angles.
- Call done only after comprehensive research is complete.
</response_protocol>"""
