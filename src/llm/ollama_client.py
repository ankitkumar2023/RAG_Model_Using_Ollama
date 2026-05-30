from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.generation_config import (
    DEFAULT_GENERATION_CONFIG,
    GUARD_GENERATION_CONFIG,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class OllamaClientError(Exception):
    """Custom exception for Ollama failures."""


class OllamaClient:
    """
    Async Ollama API client.
    """

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = settings.ollama_timeout

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.ConnectError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a completion response.
        """

        endpoint = f"{self.base_url}/api/generate"

        payload = {
            "model": model or settings.primary_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "options": options or DEFAULT_GENERATION_CONFIG,
        }

        try:
            response = await self._client.post(
                endpoint,
                json=payload,
            )

            response.raise_for_status()

            return response.json()

        except httpx.HTTPError as exc:
            logger.exception("Ollama generate request failed.")
            raise OllamaClientError(str(exc)) from exc

    async def generate_guard_response(
        self,
        prompt: str,
    ) -> str:
        """
        Run safety moderation model.
        """

        response = await self.generate(
            prompt=prompt,
            model=settings.guard_model,
            stream=False,
            options=GUARD_GENERATION_CONFIG,
        )

        return response.get("response", "").strip()

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from Ollama.
        """

        endpoint = f"{self.base_url}/api/generate"

        payload = {
            "model": model or settings.primary_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": options or DEFAULT_GENERATION_CONFIG,
        }

        try:
            async with self._client.stream(
                "POST",
                endpoint,
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)

                        token = data.get("response", "")

                        if token:
                            yield token

                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse streaming JSON chunk."
                        )

        except httpx.HTTPError as exc:
            logger.exception("Streaming request failed.")
            raise OllamaClientError(str(exc)) from exc

    async def embeddings(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """
        Generate embeddings using Ollama.
        """

        endpoint = f"{self.base_url}/api/embeddings"

        payload = {
            "model": model or settings.embedding_model,
            "prompt": text,
        }

        try:
            response = await self._client.post(
                endpoint,
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

            return data["embedding"]

        except Exception as exc:
            logger.exception("Embedding generation failed.")
            raise OllamaClientError(str(exc)) from exc

    async def health_check(self) -> bool:
        """
        Verify Ollama server availability.
        """

        try:
            response = await self._client.get(
                f"{self.base_url}/api/tags"
            )

            return response.status_code == 200

        except Exception:
            return False