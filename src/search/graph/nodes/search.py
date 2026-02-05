"""Search node implementation."""

import sys

from langchain_core.messages import ToolMessage

from ...clients.searxng import SearXNGClient
from ...config import get_settings
from ..state import SearchState


def _debug(msg: str) -> None:
    """Print debug message if debug mode is enabled."""
    if get_settings().debug:
        print(f"[DEBUG] {msg}", file=sys.stderr)


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
