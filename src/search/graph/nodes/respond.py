"""Respond node implementation."""

import sys

from ...clients.searxng import SearchResult, SearXNGClient
from ...config import get_settings
from ...llm.client import get_llm_client
from ..state import SearchState


def _debug(msg: str) -> None:
    """Print debug message if debug mode is enabled."""
    if get_settings().debug:
        print(f"[DEBUG] {msg}", file=sys.stderr)


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
