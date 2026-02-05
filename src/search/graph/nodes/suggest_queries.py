"""Suggest queries node implementation."""

import json
import re
import sys
from datetime import datetime

from langchain_core.messages import AIMessage

from ...config import get_settings
from ...llm.client import get_llm_client
from ..state import SearchState


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
