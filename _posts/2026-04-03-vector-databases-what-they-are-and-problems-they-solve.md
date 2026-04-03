---
layout: post
title: "Vector Databases: What They Are and the Problems They Solve"
date: 2026-04-03
categories: [GenAI, Data Engineering]
tags: [vector-db, embeddings, milvus, rag, llm, similarity-search]
---

If you've been working with LLMs or building RAG (Retrieval-Augmented Generation) pipelines, you've likely heard the term **Vector Database**. But what exactly is it, and why does it exist when we already have PostgreSQL, Elasticsearch, and Redis?

Let me break it down.

---

## The Problem: Traditional Databases Can't Handle "Meaning"

Traditional databases are excellent at exact lookups.

```sql
SELECT * FROM products WHERE name = 'running shoes';
```

But what if you want to find results that are *semantically similar*? For example:

> "Show me products related to marathon training gear"

A SQL `LIKE` or full-text search won't cut it — they match keywords, not **meaning**. They have no concept of the fact that "marathon training gear" is related to "running shoes", "hydration vests", or "compression socks".

This is the core problem vector databases solve.

---

## What Is a Vector Database?

A vector database stores and searches **embeddings** — high-dimensional numerical representations of data (text, images, audio, code) that encode *semantic meaning*.

When you pass a sentence through an embedding model (like OpenAI's `text-embedding-ada-002` or an open-source model like `sentence-transformers`), you get back a list of ~768 to 1536 floating point numbers — a vector. Similar content produces similar vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vec1 = model.encode("running shoes for marathon")
vec2 = model.encode("footwear for long distance running")
vec3 = model.encode("Italian pasta recipe")

# vec1 and vec2 will be close together in vector space
# vec3 will be far away
```

A vector database indexes these vectors and lets you query:

> *"Give me the top 5 most similar vectors to this query vector"*

This is called **Approximate Nearest Neighbor (ANN) search**.

---

## Problems Vector Databases Solve

### 1. Semantic Search
Find documents, products, or records by meaning — not just keyword match.

- "cheap flights to Europe" finds results about "budget airlines to Paris" even without exact word overlap.

### 2. RAG (Retrieval-Augmented Generation)
The backbone of most LLM applications today. Instead of stuffing all your data into the LLM context (expensive + limited), you:

1. Chunk your documents and embed them
2. Store embeddings in a vector DB
3. At query time, embed the user's question
4. Fetch the top-k most relevant chunks
5. Pass only those chunks to the LLM

This makes LLMs accurate on *your private data* without fine-tuning.

```
User Query → Embed → Vector DB Search → Top-K Chunks → LLM → Answer
```

### 3. Recommendation Systems
Find similar items to what a user has already liked — same concept, different domain. Netflix, Spotify, and Amazon all use variants of this.

### 4. Duplicate / Near-Duplicate Detection
Find near-duplicate records in massive datasets — useful for data deduplication pipelines.

### 5. Anomaly Detection
Vectors that are far from all other vectors in the space are potential anomalies.

---

## How ANN Search Works (Briefly)

Brute-force exact search over millions of vectors is too slow. Vector DBs use indexing algorithms like:

- **HNSW** (Hierarchical Navigable Small World) — graph-based, very fast, high recall
- **IVF** (Inverted File Index) — clusters vectors, searches only relevant clusters
- **FLAT** — exact search, used for small datasets

Most production systems use **HNSW** for its balance of speed and accuracy.

---

## Popular Vector Databases

| Database | Best For | Notes |
|---|---|---|
| **Milvus** | Large-scale production | Open-source, highly scalable, cloud-native |
| **Pinecone** | Managed/serverless | No infra to manage, easy to start |
| **Weaviate** | Hybrid search | Combines vector + keyword search |
| **Qdrant** | Rust-based, fast | Great performance, good filtering |
| **pgvector** | Already on Postgres | Add vector search to existing Postgres DB |
| **Chroma** | Local/dev use | Great for prototyping RAG pipelines |

I personally use **Milvus** in production — it handles billions of vectors well and integrates cleanly with LangChain and LangGraph pipelines.

---

## When Should You NOT Use a Vector Database?

Vector DBs are not a silver bullet. Avoid them when:

- You need **exact matches** — use a traditional DB
- Your dataset is **small** (< 100k records) — `pgvector` or even in-memory search is fine
- You need **ACID transactions** — vector DBs are not built for this

---

## Wrapping Up

Vector databases are a fundamental piece of the modern AI stack. If you're building anything with LLMs — chatbots, document Q&A, semantic search, recommendation engines — you'll likely need one.

The key insight: **traditional databases store facts, vector databases store meaning**.

---

*Have questions or want to dive deeper into Milvus internals or building RAG pipelines with LangGraph? Drop a comment or connect with me on [LinkedIn](https://www.linkedin.com/in/lalit-singh-04731a69/).*
