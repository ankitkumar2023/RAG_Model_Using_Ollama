from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from config.settings import get_settings
from src.api.schemas import ChatMessageSchema
from src.llm.gemini_client import GeminiClient
from src.llm.response_parser import ResponseParser
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

settings = get_settings()


_CONDENSE_PROMPT = """
Given the conversation so far and a follow-up question, rephrase the
follow-up into a standalone question that can be understood without
the conversation history. Do not answer the question -- only rephrase
it. If it is already standalone, return it unchanged, with no other
text.

Conversation:
{history}

Follow-up question: {question}

Standalone question:
"""

_RERANK_PROMPT = """
Given the user question and a list of candidate passages, rate how
relevant each passage is to answering the question, from 0.0 (not
relevant) to 1.0 (directly answers it).

Question: {question}

Passages:
{passages}

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"scores": [<one number per passage, same order>]}}
"""


@dataclass
class RetrievedDocument:
    """
    A single retrieved chunk, scored and ready for citation.
    """

    text: str
    score: float
    source: str
    page: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    """
    Retrieval pipeline: conversation-aware query condensing ->
    MultiQuery + MMR retrieval -> cosine-similarity scoring -> optional
    Gemini-based reranking -> score-threshold filtering.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        gemini_client: GeminiClient,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.gemini_client = gemini_client

    def _build_retriever(self, top_k: int):
        search_type = "mmr" if settings.enable_mmr else "similarity"
        search_kwargs: dict[str, Any] = {"k": top_k}

        if settings.enable_mmr:
            search_kwargs["lambda_mult"] = settings.mmr_lambda
            search_kwargs["fetch_k"] = top_k * settings.rerank_candidate_multiplier

        base_retriever = self.vector_store.store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

        if not settings.enable_multi_query:
            return base_retriever

        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.gemini_client.get_langchain_llm(temperature=0.0),
        )

    async def condense_query(
        self,
        query: str,
        chat_history: list[ChatMessageSchema] | None,
    ) -> str:
        """
        Resolve conversational references (e.g. "what about it?") into a
        standalone retrieval query, using only the last few turns so
        stale topics don't leak into unrelated retrieval.
        """

        if not chat_history:
            return query

        recent = chat_history[-settings.condense_history_turns :]
        history_text = "\n".join(f"{m.role}: {m.content}" for m in recent)

        prompt = _CONDENSE_PROMPT.format(history=history_text, question=query)

        try:
            result = await self.gemini_client.agenerate(prompt=prompt, temperature=0.0)

            condensed = result.get("response", "").strip()

            return condensed or query

        except Exception:
            logger.exception("Query condensing failed; using original query.")
            return query

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    async def _score_documents(
        self,
        query_embedding: list[float],
        documents: list,
    ) -> list[tuple]:
        ids = [doc.metadata.get("chunk_uid", "") for doc in documents]

        embeddings_by_id = await self.vector_store.get_embeddings(ids)

        scored = []

        for doc, doc_id in zip(documents, ids, strict=True):
            embedding = embeddings_by_id.get(doc_id)

            score = self._cosine_similarity(query_embedding, embedding) if embedding is not None else 0.0

            scored.append((doc, score))

        return scored

    async def _rerank(
        self,
        query: str,
        scored_docs: list[tuple],
    ) -> list[tuple]:
        if not settings.enable_reranking or not scored_docs:
            return scored_docs

        passages = "\n".join(f"[{i}] {doc.page_content[:500]}" for i, (doc, _) in enumerate(scored_docs))

        prompt = _RERANK_PROMPT.format(question=query, passages=passages)

        try:
            result = await self.gemini_client.agenerate(prompt=prompt, temperature=0.0)

            parsed = ResponseParser.extract_json(result.get("response", ""))
            scores = parsed.get("scores") if parsed else None

            if not scores or len(scores) != len(scored_docs):
                raise ValueError("Rerank score count mismatch.")

            return [(doc, float(new_score)) for (doc, _), new_score in zip(scored_docs, scores, strict=True)]

        except Exception:
            logger.exception("Reranking failed; falling back to similarity scores.")
            return scored_docs

    async def retrieve(
        self,
        query: str,
        chat_history: list[ChatMessageSchema] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve and score the most relevant chunks for a query.
        """

        top_k = top_k or settings.top_k_results

        standalone_query = await self.condense_query(query, chat_history)

        retriever = self._build_retriever(top_k)

        try:
            documents = await retriever.ainvoke(standalone_query)
        except Exception:
            logger.exception("Retrieval failed.")
            return []

        if not documents:
            return []

        query_embedding = await self.embedding_generator.embed_text(standalone_query)

        scored_docs = await self._score_documents(query_embedding, documents)
        scored_docs = await self._rerank(standalone_query, scored_docs)

        scored_docs.sort(key=lambda pair: pair[1], reverse=True)
        scored_docs = scored_docs[:top_k]

        results: list[RetrievedDocument] = []

        for doc, score in scored_docs:
            if score < settings.retrieval_score_threshold:
                continue

            results.append(
                RetrievedDocument(
                    text=doc.page_content,
                    score=score,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    metadata=doc.metadata,
                )
            )

        logger.info("Retrieved %s relevant chunks for query.", len(results))

        return results

    @staticmethod
    def build_context(retrieved_docs: list[RetrievedDocument]) -> str:
        """
        Convert retrieved chunks into LLM-ready context, with source and
        page attribution per block.
        """

        if not retrieved_docs:
            return ""

        blocks = []

        for idx, doc in enumerate(retrieved_docs, start=1):
            page_label = f", page {doc.page}" if doc.page else ""

            blocks.append(
                f"[Document {idx}]\n"
                f"Source: {doc.source}{page_label}\n"
                f"Relevance: {doc.score:.2f}\n\n"
                f"{doc.text}"
            )

        return "\n\n".join(blocks)

    async def compute_grounding_score(
        self,
        answer_text: str,
        retrieved_docs: list[RetrievedDocument],
    ) -> float:
        """
        Semantic-similarity proxy for how well the generated answer is
        actually supported by the retrieved chunks: cosine similarity
        between the answer's embedding and each chunk's embedding,
        averaged.
        """

        if not answer_text or not retrieved_docs:
            return 0.0

        answer_embedding = await self.embedding_generator.embed_text(answer_text)

        ids = [doc.metadata.get("chunk_uid", "") for doc in retrieved_docs]
        embeddings_by_id = await self.vector_store.get_embeddings(ids)

        similarities = [
            self._cosine_similarity(answer_embedding, embeddings_by_id[doc_id])
            for doc_id in ids
            if doc_id in embeddings_by_id
        ]

        if not similarities:
            return 0.0

        return sum(similarities) / len(similarities)
