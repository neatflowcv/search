"""Verify node implementation."""

import json
import re
import sys
from typing import Literal

from langgraph.types import Command

from ...clients.searxng import SearchResult, SearXNGClient
from ...config import get_settings
from ...llm.client import get_llm_client
from ..state import SearchState


def _debug(msg: str) -> None:
    """Print debug message if debug mode is enabled."""
    if get_settings().debug:
        print(f"[DEBUG] {msg}", file=sys.stderr)


async def verify_node(
    state: SearchState,
) -> Command[Literal["respond", "__end__"]]:
    """Verify the response against search results.

    Checks if the response is accurate and supported by the search results.
    If verification fails, returns to respond node with feedback.
    """
    _debug("=== verify_node ===")

    # Skip verification if already passed once
    if state.get("verification_passed"):
        _debug("Already verified, skipping")
        return Command(goto="__end__")

    llm = get_llm_client()

    # Format search results for context
    context = ""
    if state["search_results"]:
        client = SearXNGClient()
        results = [SearchResult(**r) for r in state["search_results"]]
        context = client.format_results_for_llm(results)

    system_prompt = """You are a fact-checker. Your job is to verify if the given response is accurate and supported by the search results.

Check for:
1. Factual accuracy - Does the response match the information in search results?
2. Unsupported claims - Are there claims not backed by the search results?
3. Hallucinations - Is there made-up information not present in the sources?

Search Results:
{context}

Response to verify:
{response}

Output your verification as JSON:
{{
  "passed": true/false,
  "issues": ["list of issues if any"],
  "feedback": "specific feedback for improvement if failed"
}}

Output ONLY the JSON, nothing else."""

    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                context=context or "No search results available.",
                response=state.get("response", ""),
            ),
        },
        {"role": "user", "content": state["query"]},
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, "content") else str(response)

    _debug(f"Verification response: {response_text}")

    # Parse verification result
    passed = True
    feedback = None

    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            passed = result.get("passed", True)
            if not passed:
                issues = result.get("issues", [])
                feedback = result.get("feedback", "")
                if issues:
                    feedback = f"Issues: {', '.join(issues)}. {feedback}"
    except json.JSONDecodeError:
        _debug("Failed to parse verification JSON, assuming passed")

    if passed:
        _debug("-> END (verification passed)")
        return Command(
            update={
                "verification_passed": True,
                "verification_feedback": None,
            },
            goto="__end__",
        )

    _debug(f"-> respond (verification failed: {feedback})")
    return Command(
        update={
            "verification_passed": False,
            "verification_feedback": feedback,
        },
        goto="respond",
    )
