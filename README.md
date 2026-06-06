# RAPID_PROTOTYPING

Production-grade Local Agentic RAG System using:

- Ollama
- Qwen2.5 7B
- Llama Guard 3
- Streamlit
- ChromaDB
- Async Python

---

# Features

## Core Capabilities

- Fully local LLM execution
- Guardrail-based moderation pipeline
- Retrieval-Augmented Generation (RAG)
- Tool calling
- Streaming responses
- Async-first architecture
- Production logging
- Memory management
- Document upload + indexing
- Chroma vector database

---

# Architecture

```text
User Query
    ↓
Guardrail Input Check
    ↓
Conversation Memory
    ↓
Tool Routing
    ↓
RAG Retrieval
    ↓
Qwen Reasoning
    ↓
Guardrail Output Check
    ↓
Streaming Response
```

---

# Models

| Purpose | Model |
|---|---|
| Reasoning | qwen2.5:7b |
| Guardrails | llama-guard3:8b |
| Embeddings | nomic-embed-text |

---

# Installation

## 1. Install Ollama

Download:

https://ollama.com/download

---

## 2. Pull Models

```bash
ollama pull qwen2.5:7b
ollama pull llama-guard3:8b
ollama pull nomic-embed-text
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Streamlit App

```bash
streamlit run main.py
```

---

# Project Structure

```text
RAPID_PROTOTYPING/
├── config/
├── prompts/
├── src/
├── tests/
├── examples/
└── main.py
```

---

# Security Pipeline

The system performs:

- Input moderation
- Output moderation
- Prompt injection filtering
- Jailbreak detection
- Unsafe content interception

using:

```text
llama-guard3:8b
```

---

# Development Philosophy

This project intentionally avoids:
- LangChain
- LlamaIndex
- Hidden abstractions

Everything is implemented explicitly for:
- full transparency
- performance tuning
- enterprise maintainability
- production debugging      
- solving the problem with example approach