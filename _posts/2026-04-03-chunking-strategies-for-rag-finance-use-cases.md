---
layout: post
title: "Chunking Strategies for RAG Pipelines: A Finance Data Perspective"
date: 2026-04-03
categories: [GenAI, Data Engineering]
tags: [rag, chunking, llm, finance, langchain, vector-db, embeddings]
---

Chunking is one of the most underrated decisions in a RAG pipeline. Get it wrong and your LLM gives hallucinated or incomplete answers — even with a perfect vector database and a powerful model.

In this post I'll walk through the major chunking strategies, with **finance-domain examples**: earnings calls, 10-K filings, research reports, and regulatory documents.

---

## Why Chunking Matters

LLMs have a context window limit. You can't dump an entire 200-page 10-K filing into a prompt. So you:

1. Split the document into chunks
2. Embed each chunk
3. At query time, retrieve only the relevant chunks

The quality of your answers depends heavily on **how you split**. A chunk that cuts a sentence in half, or mixes unrelated sections, will confuse the retriever and pollute the LLM context.

---

## Strategy 1: Fixed-Size Chunking

Split text into chunks of N characters (or tokens), with an optional overlap.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)

chunks = splitter.split_text(document_text)
```

**Pros:** Simple, predictable, fast.  
**Cons:** Splits mid-sentence, mid-table, mid-paragraph — loses context.

### Finance Use Case: Avoid for Structured Reports
A 10-K filing has distinct sections (Risk Factors, MD&A, Financial Statements). Fixed-size chunking will blindly split across section boundaries, mixing "Revenue grew 12%" with "Key Risk: interest rate exposure" in the same chunk.

**Verdict: Avoid for finance documents. Use only as a last resort fallback.**

---

## Strategy 2: Recursive Character Splitting

The most commonly used strategy. Splits on a hierarchy of separators (`\n\n`, `\n`, ` `, `""`) — tries to keep paragraphs together first, falls back to sentences, then words.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_text(earnings_call_transcript)
```

**Pros:** Respects natural text boundaries. Works well on most prose.  
**Cons:** Still unaware of document *structure* (sections, tables, headings).

### Finance Use Case: Earnings Call Transcripts
Earnings call transcripts are mostly conversational prose — CEO/CFO statements, analyst Q&A. Recursive splitting handles this well since it preserves speaker turns naturally.

```
Chunk 1:
"Our Q3 revenue came in at $4.2 billion, up 14% year-over-year,
driven primarily by strong performance in our cloud segment..."

Chunk 2:
"Analyst: Can you speak to the margin pressure in APAC?
CFO: Yes, we saw about 80bps of compression due to FX headwinds..."
```

**Verdict: Good default for earnings calls and financial news articles.**

---

## Strategy 3: Markdown / Header-Based Chunking

Split based on document structure — headers, sections, subsections. Each chunk stays within a logical section and inherits metadata about which section it belongs to.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "section"),
    ("##", "subsection"),
    ("###", "topic"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_report)

# Each chunk has metadata:
# {"section": "Risk Factors", "subsection": "Market Risk", "content": "..."}
```

**Pros:** Chunks are semantically coherent. Section metadata enables filtered retrieval.  
**Cons:** Requires structured input (markdown or HTML-converted PDFs).

### Finance Use Case: 10-K / Annual Reports
A 10-K has clear sections: Business Overview, Risk Factors, MD&A, Financial Statements. Header-based chunking keeps each section intact and lets you filter retrieval by section.

```python
# Query only within Risk Factors section
results = vector_db.search(
    query_vector=embed("interest rate risk"),
    filter={"section": "Risk Factors"},
    top_k=5
)
```

This is powerful — a question about revenue goes to MD&A, a question about legal exposure goes to Risk Factors. No cross-contamination.

**Verdict: Best for 10-K filings, research reports, regulatory documents.**

---

## Strategy 4: Sentence-Level Chunking

Split into individual sentences, then group N sentences together into a chunk. Gives fine-grained control.

```python
import nltk
from langchain.text_splitter import NLTKTextSplitter

nltk.download("punkt")

splitter = NLTKTextSplitter(chunk_size=3)  # 3 sentences per chunk
chunks = splitter.split_text(analyst_report)
```

Or using spaCy for better sentence boundary detection:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(analyst_report)
sentences = [sent.text.strip() for sent in doc.sents]

# Group into chunks of 3 sentences
chunk_size = 3
chunks = [
    " ".join(sentences[i:i+chunk_size])
    for i in range(0, len(sentences), chunk_size)
]
```

**Pros:** Precise boundaries, no mid-sentence cuts.  
**Cons:** May break multi-sentence context (e.g., a 5-sentence argument split across two chunks).

### Finance Use Case: Financial News & Analyst Notes
Short-form financial content like news articles or analyst commentary works well here. Each chunk captures a complete thought.

```
Chunk 1:
"HDFC Bank reported a net profit of ₹16,511 crore for Q3 FY26,
up 2.2% YoY. Net interest income grew 8% to ₹30,650 crore."

Chunk 2:
"The bank's gross NPA ratio improved to 1.24% from 1.42% a year ago.
Management guided for 15–18% loan book growth in FY27."
```

**Verdict: Good for news feeds, analyst notes, and short-form financial commentary.**

---

## Strategy 5: Semantic Chunking

The most intelligent strategy. Instead of splitting on character count or structure, it splits based on **semantic shifts** — when the topic changes, start a new chunk.

Uses embedding similarity between consecutive sentences to detect topic boundaries.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # split when similarity drops
    breakpoint_threshold_amount=90
)

chunks = splitter.split_text(sec_filing_text)
```

**Pros:** Chunks are topically coherent — the best retrieval quality.  
**Cons:** Slow (requires embedding every sentence), expensive at scale.

### Finance Use Case: SEC Filings & Regulatory Documents
SEC filings (10-Q, 8-K, proxy statements) often have dense, unstructured text where topics shift mid-paragraph. Semantic chunking handles this gracefully — it detects when the narrative shifts from "revenue discussion" to "litigation disclosure" even without explicit headers.

**Verdict: Best quality for unstructured regulatory and legal documents. Use when retrieval accuracy is critical and latency/cost is acceptable.**

---

## Strategy 6: Sliding Window Chunking

Create overlapping chunks to ensure context isn't lost at chunk boundaries.

```python
def sliding_window_chunks(text, chunk_size=600, overlap=200):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

chunks = sliding_window_chunks(credit_report_text)
```

**Pros:** No information loss at boundaries.  
**Cons:** Redundant chunks increase storage and retrieval noise.

### Finance Use Case: Credit Risk Reports
Credit reports have long continuous narratives about a borrower's financial health. A key sentence like "The borrower defaulted on a ₹50 crore loan in FY24" must not be cut across chunk boundaries. Sliding window ensures it's fully captured in at least one chunk.

**Verdict: Good safety net for long-form credit and risk reports.**

---

## Strategy Comparison at a Glance

| Strategy | Best Finance Use Case | Retrieval Quality | Cost/Speed |
|---|---|---|---|
| Fixed-Size | Avoid | Low | Fast |
| Recursive Split | Earnings calls, news | Medium | Fast |
| Header-Based | 10-K, annual reports | High | Fast |
| Sentence-Level | Analyst notes, short news | Medium-High | Fast |
| Semantic | SEC filings, legal docs | Highest | Slow/Expensive |
| Sliding Window | Credit/risk reports | Medium | Medium |

---

## A Practical Finance RAG Architecture

For a real-world finance RAG system, I'd combine strategies based on document type:

```python
def get_chunker(doc_type: str):
    if doc_type == "annual_report":
        return MarkdownHeaderTextSplitter(...)      # Header-based
    elif doc_type == "earnings_transcript":
        return RecursiveCharacterTextSplitter(...)  # Recursive
    elif doc_type == "sec_filing":
        return SemanticChunker(...)                 # Semantic
    elif doc_type == "news_article":
        return NLTKTextSplitter(...)                # Sentence-level
    elif doc_type == "credit_report":
        return sliding_window_chunks                # Sliding window
```

Route documents to the right chunker at ingestion time and tag chunks with `doc_type` metadata for filtered retrieval.

---

## Key Takeaway

> There is no universal best chunking strategy. The right choice depends on your document structure, query patterns, and latency/cost constraints.

For finance, where a single misquoted number or missed risk disclosure can matter enormously, **structured and semantic chunking will always outperform naive fixed-size splitting**.

---

*Building a finance RAG pipeline or a multi-agent data discovery system? Connect with me on [LinkedIn](https://www.linkedin.com/in/lalit-singh-04731a69/) — always happy to discuss architecture.*
