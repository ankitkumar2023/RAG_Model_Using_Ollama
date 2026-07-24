# Architecture

## Request flow

```
User query
    │
    ▼
Input safety screen (src/security/prompt_guard.py)
    │
    ▼
Query Router (src/routing/query_router.py)
    │   deterministic regex pass first (near-zero latency);
    │   falls through to one Gemini call only for ambiguous queries
    │
    ├── document_rag ──► Retriever (MultiQuery + MMR + rerank)
    ├── web_search    ──► SerpAPI
    ├── weather       ──► OpenWeatherMap
    ├── finance       ──► Alpha Vantage
    ├── hybrid        ──► Retriever + SerpAPI in parallel (asyncio.gather)
    ├── tool_calling  ──► bounded ReAct-style loop (calculator)
    └── general       ──► Gemini directly, no external context
    │
    ▼
Corrective RAG check
    (if document_rag retrieval confidence < CORRECTIVE_RAG_THRESHOLD,
     automatically merges in a web search rather than answering
     "not found")
    │
    ▼
Gemini generation (streamed token-by-token)
    │
    ▼
Citations + confidence scoring + Router Inspector diagnostics
```

## Retrieval pipeline

1. **Condense** — a follow-up question ("what about it?") is rewritten
   into a standalone query using the last few conversation turns
   (`Retriever.condense_query`), so pronoun references resolve
   correctly without dumping full chat history into the retrieval
   query itself.
2. **Multi-Query + MMR** — `langchain_classic.retrievers.multi_query.MultiQueryRetriever`
   wraps a Chroma retriever configured for Maximal Marginal Relevance
   (`search_type="mmr"`), generating several phrasings of the query and
   merging/deduplicating the results for both semantic coverage and
   result diversity.
3. **Scoring** — LangChain's retriever interface returns plain
   documents with no relevance scores attached. Scores are recomputed
   directly: each candidate's stored embedding is fetched from Chroma
   (`VectorStore.get_embeddings`, no redundant re-embedding call) and
   compared to the query embedding via cosine similarity.
4. **Reranking** — an optional lightweight Gemini call re-scores the
   candidate set against the original question (`ENABLE_RERANKING`).
   This is deliberately not a cross-encoder model — that would pull in
   a `torch`/`sentence-transformers` dependency into every deployment
   target for a marginal quality gain.
5. **Confidence score** — a blend of mean retrieval relevance and a
   grounding score (cosine similarity between the generated answer's
   embedding and the retrieved chunks' embeddings), so confidence
   reflects both "did we find relevant chunks" and "did the model
   actually use them."

## Safety

Llama Guard has no cloud equivalent, so safety is two layers:

- **App-side**: `src/security/prompt_guard.py` — a fast deterministic
  regex screen for obvious prompt-injection/jailbreak patterns, run
  before anything else.
- **Provider-side**: Gemini's own `safety_settings`
  (`src/llm/gemini_client.py`), evaluated server-side on every
  generation call.

File uploads are validated (`src/security/file_validation.py`) against
an extension allow-list and size ceiling before entering the ingestion
pipeline. Requests are rate-limited per session
(`src/security/rate_limiter.py`).

## Streaming

`APIRouter.stream_chat` runs routing and context-gathering to
completion first (that work is represented by the pipeline-stage
animation in the UI), then streams only the final generation call
token-by-token via `GeminiClient.astream`. Since the app's core is
async but Streamlit's script execution is synchronous,
`src/utils/async_bridge.py` bridges the async token generator onto a
background thread so `st.write_stream` can consume it as a plain
synchronous iterator.

## Why LangChain here, but not everywhere

`langchain-google-genai`, `langchain-chroma`, and
`langchain-classic`'s `MultiQueryRetriever` are used for the pieces
they're genuinely good at — provider integration and retrieval
composition. Scoring, reranking, routing, and orchestration are
hand-rolled in `src/`, since those are exactly the places where this
project's behavior (citations with page numbers, corrective RAG,
router diagnostics) needed to be fully visible and controllable rather
than hidden behind a chain abstraction.

## Known limitation: ephemeral storage on Streamlit Community Cloud

ChromaDB persists to `./data/vector_store` on local disk. Streamlit
Community Cloud's filesystem is **ephemeral** — it does not survive a
reboot or redeploy. Uploaded documents and their embeddings will need
to be re-indexed after the app restarts on Cloud. For a deployment
that needs persistence across restarts, point `VECTOR_DB_PATH` at a
mounted volume (self-hosted/Docker) or swap in a hosted vector store —
tracked in the README's roadmap.
