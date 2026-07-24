from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class WebSearchError(Exception):
    """Web search tool exception."""


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    link: str


class WebSearchTool:
    """
    Lightweight SERP API integration.
    """

    BASE_URL = "https://serpapi.com/search.json"

    def __init__(self) -> None:
        self.api_key = settings.serpapi_api_key

    async def execute(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """
        Perform web search.
        """

        if not self.api_key:
            raise WebSearchError("Missing SERP API key.")

        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": max_results,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                )

                response.raise_for_status()

                data = response.json()

                organic_results = data.get(
                    "organic_results",
                    [],
                )

                results: list[WebSearchResult] = []

                for item in organic_results:
                    results.append(
                        WebSearchResult(
                            title=item.get(
                                "title",
                                "",
                            ),
                            snippet=item.get(
                                "snippet",
                                "",
                            ),
                            link=item.get(
                                "link",
                                "",
                            ),
                        )
                    )

                logger.info(
                    "Web search success: %s",
                    query,
                )

                return results

            except Exception as exc:
                logger.exception("Web search failed.")

                raise WebSearchError(str(exc)) from exc
