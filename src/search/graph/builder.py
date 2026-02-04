"""LangGraph StateGraph assembly."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import research_node, respond_node, search_node
from .state import SearchState


def build_search_graph() -> CompiledStateGraph:
    """Build the search agent StateGraph.

    Graph structure:
        START -> research -> search -> research (loop)
                         -> respond -> END
    """
    # Create graph with state schema
    graph = StateGraph(SearchState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("search", search_node)
    graph.add_node("respond", respond_node)

    # Define edges
    # START -> research
    graph.add_edge(START, "research")

    # search -> research (continue loop)
    graph.add_edge("search", "research")

    # respond -> END
    graph.add_edge("respond", END)

    # Note: research node uses Command for routing to search or respond

    return graph.compile()
