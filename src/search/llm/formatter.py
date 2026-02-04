"""LFM2.5 prompt formatter with special tokens."""

import json
import re
from datetime import datetime
from typing import Literal


class LFM25Formatter:
    """Formats prompts according to LFM2.5 chat template."""

    # Special tokens
    START_OF_TEXT = "<|startoftext|>"
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    TOOL_CALL_START = "<|tool_call_start|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_LIST_START = "<|tool_list_start|>"
    TOOL_LIST_END = "<|tool_list_end|>"
    TOOL_RESPONSE_START = "<|tool_response_start|>"
    TOOL_RESPONSE_END = "<|tool_response_end|>"

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
                            "description": "List of search queries (max 3)",
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

    def format_user_message(self, content: str) -> str:
        """Format user message with LFM2.5 tokens."""
        return (
            f"{self.IM_START}user\n{content}{self.IM_END}\n{self.IM_START}assistant\n"
        )

    def format_tool_response(self, response: str) -> str:
        """Format tool response with LFM2.5 tokens."""
        return (
            f"{self.IM_START}tool\n{response}{self.IM_END}\n{self.IM_START}assistant\n"
        )

    def parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from assistant response.

        Extracts Pythonic function calls from between TOOL_CALL_START and TOOL_CALL_END.
        Example: [web_search(queries=["query1", "query2"])]
        """
        tool_calls = []

        # Find content between tool call markers
        pattern = (
            re.escape(self.TOOL_CALL_START) + r"(.*?)" + re.escape(self.TOOL_CALL_END)
        )
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            # Parse individual function calls
            # Pattern: function_name(param="value", ...)
            func_pattern = r"(\w+)\((.*?)\)"
            func_matches = re.findall(func_pattern, match, re.DOTALL)

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
                            if param_value.startswith('"') and param_value.endswith(
                                '"'
                            ):
                                tool_call["arguments"][param_name] = param_value[1:-1]
                            else:
                                tool_call["arguments"][param_name] = param_value

                tool_calls.append(tool_call)

        return tool_calls

    def _get_speed_prompt(
        self, tool_desc: str, iteration: int, max_iterations: int, today: str
    ) -> str:
        """Generate speed mode system prompt."""
        return f"""{self.START_OF_TEXT}{self.IM_START}system
You are an action orchestrator. Your job is to fulfill user requests by selecting and executing the available tools—no free-form replies.

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
- Default to web_search when information is missing or stale; keep queries targeted (max 3 per call).
- Call done when you have gathered enough to answer or performed the required actions.
</response_protocol>{self.IM_END}"""

    def _get_balanced_prompt(
        self, tool_desc: str, iteration: int, max_iterations: int, today: str
    ) -> str:
        """Generate balanced mode system prompt."""
        return f"""{self.START_OF_TEXT}{self.IM_START}system
You are an action orchestrator. Your job is to fulfill user requests by reasoning briefly and executing the available tools—no free-form replies.

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
- Default to web_search when information is missing or stale; keep queries targeted (max 3 per call).
- Call done only after you have the needed info or actions completed.
</response_protocol>{self.IM_END}"""

    def _get_quality_prompt(
        self, tool_desc: str, iteration: int, max_iterations: int, today: str
    ) -> str:
        """Generate quality mode system prompt."""
        return f"""{self.START_OF_TEXT}{self.IM_START}system
You are a deep-research orchestrator. Your job is to fulfill user requests with thorough, comprehensive research—no free-form replies.

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
</response_protocol>{self.IM_END}"""
