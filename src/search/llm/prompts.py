"""Prompt formatter for LLM."""

import json
from datetime import datetime
from typing import Literal

from .parser import TOOL_CALL_END, TOOL_CALL_START, parse_tool_calls
from .tools import get_tools_definition

# Tool list tokens
TOOL_LIST_START = "<|tool_list_start|>"
TOOL_LIST_END = "<|tool_list_end|>"


class PromptFormatter:
    """Formats prompts for LLM chat template.

    Note: Chat template tokens (<|im_start|>, <|im_end|>, etc.) are NOT included
    in the output. The llama-server applies these automatically from the GGUF
    model's embedded chat template.
    """

    def __init__(
        self, mode: Literal["speed", "balanced", "quality"] = "balanced"
    ) -> None:
        self.mode = mode

    def get_tools_definition(self) -> list[dict]:
        """Get tool definitions for the search agent."""
        return get_tools_definition(self.mode)

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
        """Parse tool calls from assistant response."""
        return parse_tool_calls(response)

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

{TOOL_LIST_START}
{tool_desc}
{TOOL_LIST_END}

<response_protocol>
- NEVER output normal text to the user. ONLY call tools using {TOOL_CALL_START} and {TOOL_CALL_END} tokens.
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

{TOOL_LIST_START}
YOU MUST CALL __reasoning_preamble BEFORE EVERY TOOL CALL IN THIS ASSISTANT TURN.
{tool_desc}
{TOOL_LIST_END}

<response_protocol>
- NEVER output normal text to the user. ONLY call tools using {TOOL_CALL_START} and {TOOL_CALL_END} tokens.
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

{TOOL_LIST_START}
YOU MUST CALL __reasoning_preamble BEFORE EVERY TOOL CALL IN THIS ASSISTANT TURN.
{tool_desc}
{TOOL_LIST_END}

<research_strategy>
For any topic, consider searching:
1. Core definition/overview - What is it?
2. Features/capabilities - What can it do?
3. Comparisons - How does it compare to alternatives?
4. Recent news/updates - What's the latest?
5. Reviews/opinions - What do experts say?
</research_strategy>

<response_protocol>
- NEVER output normal text to the user. ONLY call tools using {TOOL_CALL_START} and {TOOL_CALL_END} tokens.
- Follow an iterative loop: __reasoning_preamble → tool call → __reasoning_preamble → tool call → ... → done.
- Aim for 4-7 information-gathering calls covering different angles.
- Call done only after comprehensive research is complete.
</response_protocol>"""
