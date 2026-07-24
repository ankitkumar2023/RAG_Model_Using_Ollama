from __future__ import annotations

RAG_QUERY_REWRITE_PROMPT = """
Rewrite the user query into an optimized semantic retrieval query.

Goals:
- preserve meaning
- improve retrieval quality
- remove ambiguity
- expand abbreviations if useful

Return only the rewritten query.
"""

DOCUMENT_SUMMARY_PROMPT = """
Generate a concise document summary.

Requirements:
- preserve key facts
- preserve technical meaning
- avoid hallucinations
- maintain factual grounding
"""

CONTEXT_COMPRESSION_PROMPT = """
Compress the retrieved context while preserving:
- factual information
- technical accuracy
- named entities
- important numerical values

Remove redundancy.
"""

ANSWER_SYNTHESIS_PROMPT = """
Generate a final grounded response using retrieved context.

Requirements:
1. Be factually accurate.
2. Use retrieved information.
3. Do not hallucinate.
4. Clearly state uncertainty if needed.
5. Keep technical precision.
"""
