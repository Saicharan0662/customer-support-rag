# Customer Support RAG System

A retrieval-augmented generation system that answers customer support questions from a document corpus. Built with LangChain, FAISS, sentence-transformers, and Flask.

## What It Does

You feed in support docs (FAQs, policies, guides). The system indexes them once. Then it answers user questions by retrieving the most relevant chunks and generating a grounded response with source citations.

## Architecture
```
User Question
     │
     ▼
Flask API (POST /query)
     │
     ▼
Embed query → 384-dim vector
     │
     ▼
FAISS retrieves top-k chunks
     │
     ▼
Send chunks + question to Llama via Groq
     │
     ▼
Return answer + sources as JSON
```

## Stack

- Python 3.10+
- LangChain for orchestration
- FAISS for vector similarity search
- sentence-transformers for local embeddings (all-MiniLM-L6-v2, 384 dimensions)
- Groq API for LLM generation (Llama 3.3 70B)
- Flask for the API layer

Add your docs to `data/docs/`. The sample set covers refunds, shipping, account setup, order tracking, and payment billing.

## Build the Index

```bash
python build_index.py
```

This loads all docs from `data/docs/`, splits them into 500-character chunks with 50-character overlap, embeds each chunk locally, and saves a FAISS index to `index/`. Takes about 30 seconds on first run (downloads the embedding model). Subsequent runs are under 5 seconds.

## Run the API

```bash
python app.py
```

The server starts on port 8080. The embedding model and FAISS index load once at startup.

## Usage

Query the API:

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I request a refund?"}'
```

Response:

```json
{
  "answer": "To request a refund, log in to your ACME account, go to Order History, select the order, and click Request Refund...\nrefund_policy.txt",
  "sources": [
    {
      "source": "data/docs/refund_policy.txt",
      "score": 0.66,
      "preview": "How to Request a Refund\nLog in to your ACME account..."
    }
  ]
}
```

### Optional Parameters

- `k` (int, default 4): number of chunks to retrieve
- `strategy` (str, default "similarity"): "similarity" or "mmr"

MMR (Maximum Marginal Relevance) returns chunks that are both relevant and diverse, useful when similarity search returns near-duplicate chunks.

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I request a refund?", "strategy": "mmr", "k": 5}'
```

## Design Decisions

**Local embeddings.** Using sentence-transformers instead of OpenAI embeddings keeps the indexing pipeline free and offline. The all-MiniLM-L6-v2 model produces 384-dimensional vectors and runs fast on CPU.

**FAISS for retrieval.** FAISS handles millions of vectors with low latency and runs in-process. No separate vector DB service needed for this scale.

**Groq for generation.** Free tier with high rate limits (30 req/min) and fast inference. The retrieval pipeline is provider-agnostic, so swapping to Anthropic, OpenAI, or a local Ollama model is a one-file change in `src/generate.py`.

**System prompt grounding.** The LLM is instructed to answer only from provided context and refuse when context is insufficient. This prevents hallucination on out-of-scope queries.

**Score-based filtering (optional).** FAISS L2 distance scores correlate with relevance. In-scope queries typically score under 1.1. Out-of-scope queries score above 1.8. A threshold can short-circuit obviously bad retrieval without an LLM call.

## Failure Modes and Mitigations

- **Off-topic retrieval:** retune `chunk_size` (try 800 or 1200 for denser docs)
- **Refusal on in-scope questions:** increase `k` from 4 to 6 or 8
- **Vocabulary mismatch between query and docs:** add a query rewriting step before retrieval
- **Exact-term queries (product codes, error IDs):** add hybrid search (FAISS + BM25) via LangChain's EnsembleRetriever
