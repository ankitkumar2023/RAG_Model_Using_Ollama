# ✨ AI Workspace

**An enterprise-grade, Gemini-powered agentic RAG platform.**

Document Q&A, live web search, weather, and finance — routed
automatically by an intelligent query router, with corrective RAG,
multi-query + MMR retrieval, source citations, and a fully custom
design system, all running on Streamlit.

---

## Screenshots

> Add screenshots to [`assets/`](assets/) and they'll render here —
> see [`assets/README.md`](assets/README.md) for the expected filenames.

| Chat | Router Inspector |
|---|---|
| `assets/chat.png` | `assets/router-inspector.png` |

| Knowledge Base | Diagnostics |
|---|---|
| `assets/knowledge-base.png` | `assets/diagnostics.png` |

---

## Features

**Retrieval & Reasoning**
- Query router — classifies every request (document Q&A, web search,
  weather, finance, hybrid, tool-calling, or general knowledge) before
  any retrieval happens, instead of always defaulting to RAG
- Corrective RAG — automatically falls back to a web search when
  retrieval confidence is low, rather than answering "not found"
- Multi-Query + MMR retrieval, with optional Gemini-based reranking
- Conversation-aware query condensing (resolves "what about it?"
  style follow-ups using recent chat history)
- Confidence scoring (retrieval relevance blended with answer-grounding
  similarity)
- Source citations with page numbers
- Bounded ReAct-style tool-calling loop (calculator, extensible)

**Chat Experience**
- True token-by-token streaming
- Markdown, code blocks, tables
- Copy / regenerate per message
- Router Inspector — a live, per-response timeline of what the router
  decided and why, with latency and confidence breakdown
- Developer Mode diagnostics dashboard

**Platform**
- Two-layer safety: a deterministic prompt-injection screen plus
  Gemini's own `safety_settings`
- File-upload validation and per-session rate limiting
- Custom glassmorphic design system (`ui/theme.py`, `ui/components.py`)
  — no default Streamlit styling
- Works locally via `.env` or on Streamlit Community Cloud via
  `st.secrets`, unchanged

---

## Architecture

```
User query
    │
    ▼
Input safety screen
    │
    ▼
Query Router ──┬─ document_rag ──► Multi-Query + MMR + rerank
               ├─ web_search   ──► SerpAPI
               ├─ weather      ──► OpenWeatherMap
               ├─ finance      ──► Alpha Vantage
               ├─ hybrid       ──► RAG + web search (parallel)
               ├─ tool_calling ──► bounded tool loop
               └─ general      ──► Gemini directly
    │
    ▼
Corrective RAG (low-confidence retrieval → web fallback)
    │
    ▼
Gemini (streamed) ──► Citations + confidence + Router Inspector
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
retrieval pipeline, safety model, and streaming design.

---

## Installation

Requires Python 3.11 or 3.12.

```bash
git clone <this-repo-url>
cd Rapid_Prototyping

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set GEMINI_API_KEY at minimum
# (get one at https://aistudio.google.com/apikey)

streamlit run main.py
```

The weather, finance, and web-search tools are optional — the app runs
fine without `OPENWEATHER_API_KEY` / `ALPHA_VANTAGE_API_KEY` /
`SERPAPI_API_KEY`; the router simply won't be able to fulfill those
specific routes.

---

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub (a plain `.env` is gitignored and
   never committed — see [Security](#security)).
2. On [share.streamlit.io](https://share.streamlit.io), create a new
   app pointing at this repo, with `main.py` as the entry point.
3. In the app's **Advanced settings → Secrets**, paste the same
   key/value pairs from `.env.example` (filled in), e.g.:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   SERPAPI_API_KEY = "your-key-here"
   ```

   `config/settings.py` reads `st.secrets` and bridges it into the
   process environment automatically — no code changes needed between
   local and Cloud.
4. Deploy.

**Note on persistence**: Streamlit Community Cloud's filesystem is
ephemeral. The local ChromaDB vector store (`data/vector_store/`) does
not survive a reboot or redeploy — documents will need to be
re-uploaded after the app restarts. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#known-limitation-ephemeral-storage-on-streamlit-community-cloud).

### Docker / self-hosted

Any environment that can run `streamlit run main.py` with the
dependencies in `requirements.txt` and a `GEMINI_API_KEY` in its
environment works. Mount a persistent volume at `data/vector_store` if
you need the index to survive restarts.

---

## Technologies

| Layer | Choice |
|---|---|
| LLM | Google Gemini (`gemini-flash-latest` / `gemini-pro-latest`) |
| Orchestration | LangChain (`langchain-google-genai`, `langchain-chroma`, `langchain-classic`) |
| Vector store | ChromaDB |
| Embeddings | Gemini Embeddings |
| UI | Streamlit + a custom CSS/HTML design system |
| Language | Python 3.11+ |

---

## Folder Structure

```
Rapid_Prototyping/
├── main.py                  # Streamlit entrypoint
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml          # base theme
├── config/                  # settings, model config, logging
├── prompts/                 # system/router prompt templates
├── src/
│   ├── api/                 # request/response schemas + orchestration (routes.py)
│   ├── llm/                 # Gemini client, response parsing
│   ├── rag/                 # chunking, embeddings, retrieval
│   ├── routing/             # query router
│   ├── security/            # prompt guard, file validation, rate limiting
│   ├── tools/                # calculator, weather, stock, web search
│   ├── utils/                # async bridge, system monitor, logging
│   └── examples/             # small runnable usage examples
├── ui/                       # design system: theme tokens + components
├── tests/
├── docs/
│   └── ARCHITECTURE.md
├── assets/                   # README screenshots
└── data/
    └── vector_store/         # ChromaDB persistence (gitignored)
```

---

## Testing

```bash
pytest tests/ -q
```

Tests mock all external calls (Gemini, weather/finance/search APIs) —
no API keys or network access required to run the suite.

---

## Security

- No secrets are committed. `.env` is gitignored; `.env.example`
  documents every variable with empty/placeholder values.
- On Streamlit Community Cloud, secrets live in the platform's Secrets
  manager (`st.secrets`), never in the repo.
- Two-layer input/output safety (see [Architecture](#architecture)).
- Uploaded files are validated for type and size before ingestion.
- Requests are rate-limited per session.

---

## Future Roadmap

- Hosted vector store option, for persistence across Streamlit Cloud
  redeploys
- Multi-step (not just single-call) tool-calling loop with additional
  tools
- Per-message pin / edit / share / export
- Suggested prompts and keyboard shortcuts
- Optional cross-encoder reranking for self-hosted deployments where a
  heavier dependency footprint is acceptable

---

## License

[MIT](LICENSE)
