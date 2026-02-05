"""LangGraph StateGraph assembly."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes.research import research_node
from .nodes.respond import respond_node
from .nodes.search import search_node
from .nodes.suggest_queries import suggest_queries_node
from .state import SearchState


def build_search_graph() -> CompiledStateGraph:
    """Build the search agent StateGraph.

    Graph structure:
        START -> suggest_queries -> search -> research -> search (loop)
                                                       -> respond -> END
    """
    # Create graph with state schema
    graph = StateGraph(SearchState)

    # Add nodes
    graph.add_node("suggest_queries", suggest_queries_node)
    graph.add_node("research", research_node)
    graph.add_node("search", search_node)
    graph.add_node("respond", respond_node)

    # Define edges
    # START -> suggest_queries (analyze query and suggest search terms)
    graph.add_edge(START, "suggest_queries")

    # suggest_queries -> search (always perform initial search)
    graph.add_edge("suggest_queries", "search")

    # search -> research (continue to research node)
    graph.add_edge("search", "research")

    # respond -> END
    graph.add_edge("respond", END)

    # Note: research node uses Command for routing to search or respond

    return graph.compile()
