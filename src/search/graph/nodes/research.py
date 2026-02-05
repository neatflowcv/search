"""Research node implementation."""

import sys
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command

from ...clients.searxng import SearchResult, SearXNGClient
from ...config import get_settings
from ...llm.client import get_llm_client
from ...llm.prompts import PromptFormatter
from ..state import SearchState


def _debug(msg: str) -> None:
    """Print debug message if debug mode is enabled."""
    if get_settings().debug:
        print(f"[DEBUG] {msg}", file=sys.stderr)


async def research_node(
    state: SearchState,
) -> Command[Literal["search", "respond"]]:
    """Main research node that decides next action.

    Calls the LLM to determine whether to search or respond.
    """
    _debug(f"=== research_node (iteration {state['iteration'] + 1}) ===")

    formatter = PromptFormatter(mode=state["mode"])
    llm = get_llm_client()

    # Build system prompt
    system_prompt = formatter.format_system_prompt(
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )

    # Build messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]},
    ]

    # Add previous search results if any
    if state["search_results"]:
        client = SearXNGClient()
        results = [SearchResult(**r) for r in state["search_results"]]
        formatted_results = client.format_results_for_llm(results)
        messages.append({
            "role": "tool",
            "content": formatted_results,
            "tool_call_id": "web_search",
        })

    _debug(f"Sending {len(messages)} messages to LLM")

    # Call LLM
    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    _debug(f"LLM response: {response_text[:200]}...")

    # Parse tool calls
    tool_calls = formatter.parse_tool_calls(response_text)

    _debug(f"Parsed tool calls: {[tc['name'] for tc in tool_calls]}")

    # Extract reasoning if present
    reasoning = []
    for tc in tool_calls:
        if tc["name"] == "__reasoning_preamble":
            thought = tc.get("arguments", {}).get("thought", "")
            if thought:
                reasoning.append(thought)
                _debug(f"Reasoning: {thought}")

    # Check if done
    is_done = any(tc["name"] == "done" for tc in tool_calls)

    if is_done or state["iteration"] >= state["max_iterations"] - 1:
        _debug("-> respond (done or max iterations)")
        return Command(
            update={
                "messages": [AIMessage(content=response_text)],
                "reasoning": state["reasoning"] + reasoning,
                "is_complete": True,
            },
            goto="respond",
        )

    # Check for web_search calls
    search_calls = [tc for tc in tool_calls if tc["name"] == "web_search"]
    if search_calls:
        queries = [
            q
            for tc in search_calls
            for q in tc.get("arguments", {}).get("queries", [])
        ]
        _debug(f"-> search (queries: {queries})")
        return Command(
            update={
                "messages": [AIMessage(content=response_text)],
                "reasoning": state["reasoning"] + reasoning,
                "pending_tool_calls": search_calls,
            },
            goto="search",
        )

    # No actionable tool calls, go to respond
    _debug("-> respond (no actionable tool calls)")
    return Command(
        update={
            "messages": [AIMessage(content=response_text)],
            "reasoning": state["reasoning"] + reasoning,
            "is_complete": True,
        },
        goto="respond",
    )
