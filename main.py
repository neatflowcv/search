"""Search agent entry point."""

import argparse
import asyncio

from src.search.config import get_settings
from src.search.graph.builder import build_search_graph
from src.search.graph.state import SearchState


async def run_search(query: str) -> str:
    """Execute a search query through the LangGraph agent.

    Args:
        query: The user's search query

    Returns:
        The agent's response
    """
    settings = get_settings()
    graph = build_search_graph()

    initial_state: SearchState = {
        "query": query,
        "messages": [],
        "search_results": [],
        "iteration": 0,
        "max_iterations": settings.max_iterations,
        "mode": settings.research_mode,
        "reasoning": [],
        "response": None,
        "is_complete": False,
        "pending_tool_calls": [],
    }

    result = await graph.ainvoke(initial_state)
    return result.get("response", "No response generated.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Search Agent (LangGraph + LFM2.5 + SearXNG)")
    parser.add_argument("query", help="검색할 질의")
    args = parser.parse_args()

    print("Search Agent (LangGraph + LFM2.5 + SearXNG)")
    print("-" * 50)
    print(f"Query: {args.query}")
    print("\nSearching...\n")

    response = asyncio.run(run_search(args.query))

    print("-" * 50)
    print("Response:")
    print("-" * 50)
    print(response)


if __name__ == "__main__":
    main()
