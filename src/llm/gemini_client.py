from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class GeminiClientError(Exception):
    """Custom exception for Gemini failures."""


# Llama Guard has no cloud equivalent -- Gemini's own safety_settings
# evaluate every generation call server-side, forming one layer of the
# two-layer input/output safety approach (the other being
# src.security.prompt_guard's deterministic pre-screen).
_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


class GeminiClient:
    """
    Thin wrapper around LangChain's `ChatGoogleGenerativeAI`, providing
    sync/async and streaming/non-streaming generation against Gemini.
    """

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise GeminiClientError("GEMINI_API_KEY is not configured.")

        self.api_key = settings.gemini_api_key.get_secret_value()

        # Keyed by (model, temperature, top_p, top_k, max_output_tokens).
        # `GeminiClient` is a long-lived, process-wide singleton (cached
        # via st.cache_resource in main.py), so caching the underlying
        # ChatGoogleGenerativeAI/async HTTP client here means each
        # distinct parameter combination is created once and reused --
        # not recreated and abandoned on every single call. Without
        # this, each throwaway client's async session was only ever
        # closed via __del__-triggered garbage collection, which is
        # racy in asyncio and was logging "Task was destroyed but it is
        # pending!" after every request.
        self._llm_cache: dict[tuple, ChatGoogleGenerativeAI] = {}

    def _get_llm(
        self,
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> ChatGoogleGenerativeAI:
        model_name = model or settings.primary_model
        temperature = generation_kwargs.get("temperature", settings.temperature)
        top_p = generation_kwargs.get("top_p", settings.top_p)
        top_k = generation_kwargs.get("top_k", settings.top_k)
        max_output_tokens = generation_kwargs.get("max_output_tokens", settings.max_tokens)

        cache_key = (model_name, temperature, top_p, top_k, max_output_tokens)

        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            safety_settings=_SAFETY_SETTINGS,
            timeout=settings.gemini_timeout,
            max_retries=settings.gemini_max_retries,
        )

        self._llm_cache[cache_key] = llm

        return llm

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: str | None,
    ) -> list[SystemMessage | HumanMessage]:
        messages: list[SystemMessage | HumanMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        return messages

    def get_langchain_llm(
        self,
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> ChatGoogleGenerativeAI:
        """
        Expose the underlying LangChain chat model, for callers (e.g.
        MultiQueryRetriever) that need a `BaseLanguageModel` directly.
        """

        return self._get_llm(model=model, **generation_kwargs)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a complete (non-streamed) response.
        """

        llm = self._get_llm(model=model, **generation_kwargs)
        messages = self._build_messages(prompt, system_prompt)

        try:
            result = await llm.ainvoke(messages)

            return {
                "response": result.text,
                "model": llm.model,
            }

        except Exception as exc:
            logger.exception("Gemini generation failed.")
            raise GeminiClientError(str(exc)) from exc

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Synchronous generation, for use inside Streamlit's script thread.
        """

        llm = self._get_llm(model=model, **generation_kwargs)
        messages = self._build_messages(prompt, system_prompt)

        try:
            result = llm.invoke(messages)

            return {
                "response": result.text,
                "model": llm.model,
            }

        except Exception as exc:
            logger.exception("Gemini generation failed.")
            raise GeminiClientError(str(exc)) from exc

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> Iterator[str]:
        """
        Synchronously stream response chunks, token-by-token.
        """

        llm = self._get_llm(model=model, **generation_kwargs)
        messages = self._build_messages(prompt, system_prompt)

        try:
            for chunk in llm.stream(messages):
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.exception("Gemini streaming failed.")
            raise GeminiClientError(str(exc)) from exc

    async def astream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **generation_kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Asynchronously stream response chunks, token-by-token.
        """

        llm = self._get_llm(model=model, **generation_kwargs)
        messages = self._build_messages(prompt, system_prompt)

        try:
            async for chunk in llm.astream(messages):
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.exception("Gemini streaming failed.")
            raise GeminiClientError(str(exc)) from exc

    def health_check(self) -> bool:
        """
        Verify a Gemini API key is configured.

        Deliberately does not make a network call -- this runs on every
        page load, and burning a real API call just to render a status
        dot would eat into the (fairly tight) free-tier daily quota.
        """

        return bool(settings.gemini_api_key)
