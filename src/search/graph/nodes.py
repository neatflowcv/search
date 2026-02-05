"""LangGraph node implementations."""

import json
import re
import sys
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from ..clients.searxng import SearXNGClient
from ..config import get_settings
from ..llm.client import get_llm_client
from ..llm.formatter import PromptFormatter
from .state import SearchState


def _debug(msg: str) -> None:
    """Print debug message if debug mode is enabled."""
    if get_settings().debug:
        print(f"[DEBUG] {msg}", file=sys.stderr)


async def suggest_queries_node(state: SearchState) -> dict:
    """Analyze user query and suggest search queries.

    This node runs first to generate optimized search queries
    from the user's input, then proceeds to web search unconditionally.
    """
    _debug("=== suggest_queries_node ===")
    _debug(f"User query: {state['query']}")

    llm = get_llm_client()
    today = datetime.now().strftime("%B %d, %Y")

    system_prompt = f"""You are a search query optimizer. Your task is to analyze the user's question and generate effective search queries.

Today's date: {today}

Given the user's question, generate 3-10 search queries that will help find the most relevant and comprehensive information.

Guidelines:
- Generate at least 3 queries, up to 10 queries for complex topics
- Cover different aspects and angles of the question
- Use specific, targeted keywords
- Include variations: definitions, comparisons, recent updates, expert opinions, use cases
- Consider including recent/latest if the topic may have updates
- Output ONLY a JSON array of query strings, nothing else

Example output:
["what is X", "X vs Y comparison", "X latest news 2024", "X best practices", "X tutorial"]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]},
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    _debug(f"LLM response: {response_text}")

    # Parse JSON array from response
    queries = []
    try:
        # Try to find JSON array in response
        json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # Fallback to original query if parsing fails
    if not queries:
        queries = [state["query"]]

    _debug(f"Suggested queries: {queries}")

    # Set up pending tool calls for search_node
    return {
        "suggested_queries": queries,
        "pending_tool_calls": [{"name": "web_search", "arguments": {"queries": queries}}],
        "messages": [AIMessage(content=f"Suggested search queries: {queries}")],
    }


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
        from ..clients.searxng import SearchResult

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


async def search_node(state: SearchState) -> dict:
    """Execute web search via SearXNG.

    Processes pending tool calls and returns search results.
    """
    _debug("=== search_node ===")

    client = SearXNGClient()

    # Extract queries from pending tool calls
    queries: list[str] = []
    for tc in state.get("pending_tool_calls", []):
        if tc["name"] == "web_search":
            tc_queries = tc.get("arguments", {}).get("queries", [])
            queries.extend(tc_queries)

    if not queries:
        queries = [state["query"]]

    _debug(f"Searching for: {queries}")

    # Execute search
    results = await client.search(queries=queries)

    _debug(f"Got {len(results)} results")

    # Format results for message
    formatted = client.format_results_for_llm(results)

    return {
        "search_results": [r.model_dump() for r in results],
        "messages": [ToolMessage(content=formatted, tool_call_id="web_search")],
        "iteration": state["iteration"] + 1,
        "pending_tool_calls": [],
    }


async def respond_node(state: SearchState) -> dict:
    """Generate final response based on search results.

    Uses the collected information to generate a comprehensive response.
    """
    _debug("=== respond_node ===")
    _debug(f"Total search results: {len(state['search_results'])}")

    llm = get_llm_client()

    # Format context from search results
    context = ""
    if state["search_results"]:
        client = SearXNGClient()
        from ..clients.searxng import SearchResult

        results = [SearchResult(**r) for r in state["search_results"]]
        context = client.format_results_for_llm(results)

    # Build response prompt
    system_prompt = """You are a helpful research assistant. Based on the search results provided, generate a comprehensive and accurate response to the user's query.

Guidelines:
- Use the search results to provide factual, up-to-date information
- Cite sources when possible by mentioning the source
- Be concise but thorough
- If the search results are insufficient, acknowledge the limitations

Search Results:
{context}"""

    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                context=context or "No search results available."
            ),
        },
        {"role": "user", "content": state["query"]},
    ]

    # Add reasoning context if available
    if state["reasoning"]:
        reasoning_summary = "\n".join(f"- {r}" for r in state["reasoning"])
        messages[0]["content"] += f"\n\nResearch reasoning:\n{reasoning_summary}"

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    return {
        "response": response_text,
        "is_complete": True,
    }
