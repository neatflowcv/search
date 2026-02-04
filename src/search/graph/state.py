"""LangGraph state definitions."""

from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage


class SearchState(TypedDict):
    """Central state object for the search graph."""

    # User query
    query: str

    # Accumulated messages (uses add reducer for append)
    messages: Annotated[list[BaseMessage], add]

    # Search results from SearXNG
    search_results: list[dict]

    # Current iteration count
    iteration: int

    # Maximum iterations allowed
    max_iterations: int

    # Research mode
    mode: Literal["speed", "balanced", "quality"]

    # Reasoning thoughts (for balanced/quality modes)
    reasoning: list[str]

    # Final response
    response: str | None

    # Whether search is complete
    is_complete: bool

    # Pending tool calls to execute
    pending_tool_calls: list[dict]
