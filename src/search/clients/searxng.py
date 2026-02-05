"""SearXNG API client."""

import httpx
from pydantic import BaseModel

from ..config import get_settings


class SearchResult(BaseModel):
    """Single search result from SearXNG."""

    title: str
    url: str
    content: str
    engine: str
    score: float | None = None


class SearXNGClient:
    """Async client for SearXNG JSON API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.searxng_url).rstrip("/")
        self.timeout = timeout or settings.searxng_timeout

    async def search(
        self,
        queries: list[str],
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        language: str = "en",
        max_results_per_query: int = 5,
    ) -> list[SearchResult]:
        """Execute search queries against SearXNG.

        Args:
            queries: List of search queries to execute
            categories: Optional category filter (e.g., ["general", "news"])
            engines: Optional engine filter (e.g., ["google", "bing"])
            language: Search language (default: "en")
            max_results_per_query: Maximum results per query

        Returns:
            List of SearchResult objects
        """
        results: list[SearchResult] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for query in queries[:10]:  # Limit to 10 queries
                params: dict[str, str] = {
                    "q": query,
                    "format": "json",
                    "language": language,
                }
                if categories:
                    params["categories"] = ",".join(categories)
                if engines:
                    params["engines"] = ",".join(engines)

                try:
                    response = await client.get(
                        f"{self.base_url}/search",
                        params=params,
                    )
                    response.raise_for_status()
                    data = response.json()

                    for item in data.get("results", [])[:max_results_per_query]:
                        results.append(
                            SearchResult(
                                title=item.get("title", ""),
                                url=item.get("url", ""),
                                content=item.get("content", ""),
                                engine=item.get("engine", ""),
                                score=item.get("score"),
                            )
                        )
                except httpx.HTTPError:
                    # Continue with other queries on error
                    continue

        return results

    def format_results_for_llm(self, results: list[SearchResult]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No search results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.title}\n"
                f"    URL: {r.url}\n"
                f"    {r.content[:300]}{'...' if len(r.content) > 300 else ''}"
            )

        return "\n\n".join(formatted)
