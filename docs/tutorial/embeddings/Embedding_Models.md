
# Embedding Models Explained
*A practical guide using BAAI BGE models as examples*

---

## 1. What Is an Embedding Model?

An **embedding model** converts text (or other data) into a fixed-length numerical vector that represents its **semantic meaning**.

- Similar texts → vectors close together
- Dissimilar texts → vectors far apart

Embedding models **do not generate text**.  
They are used for *understanding, matching, and retrieval*.

---

## 2. Embedding Models vs Large Language Models (LLMs)

| Feature | Embedding Model | LLM |
|------|----------------|-----|
| Output | Vector (numbers) | Text |
| Purpose | Similarity & retrieval | Generation & reasoning |
| Training objective | Contrastive / similarity | Next-token prediction |
| Example | `bge-small-zh` | GPT, LLaMA |
| Used in RAG | Retriever | Generator |

Embedding models are often **paired with LLMs**, not replaced by them.

---

## 3. Example: BAAI BGE Models

### bge-small-zh
- 384-dimensional embeddings
- Lightweight, fast
- Lower VRAM & storage cost
- Best for large-scale or latency-sensitive systems

### bge-large-zh-v1.5
- 1024-dimensional embeddings
- Higher semantic precision
- Higher VRAM & index cost
- Best for high-quality retrieval

---

## 4. Core Technical Concepts

### 4.1 Embedding Dimension
- Size of the output vector (e.g., 384, 768, 1024)
- Higher = more expressive, but:
  - More VRAM
  - Larger vector databases
  - Slower similarity search

### 4.2 Vector Similarity
Common metrics:
- **Cosine similarity** (most common)
- Dot product
- Euclidean distance

Cosine similarity focuses on *direction*, not magnitude.

---

## 5. How Embedding Models Are Trained

Most modern embedding models use **contrastive learning**:

- Positive pairs: (query, relevant text)
- Negative pairs: (query, unrelated text)

Goal:
> Pull related texts closer, push unrelated texts apart in vector space.

This is why embedding models are excellent for search and matching.

---

## 6. Memory & VRAM Considerations

VRAM usage depends on:

```
Model parameters
+ Activations (batch_size × sequence_length)
+ Precision (FP32 vs FP16)
```

### Key takeaways:
- Smaller models can use MORE VRAM if batch size is larger
- Sequence length has quadratic memory cost (attention)
- FP16 / BF16 halves memory vs FP32

---

## 7. Common Practical Issues

### 7.1 Batch Size Inflation
Small models are often run with very large batches → high VRAM usage.

### 7.2 Precision Mismatch
FP32 inference can double memory usage unintentionally.

### 7.3 Token Length
Long documents (512–1024 tokens) dramatically increase memory.

### 7.4 Poor Pooling Strategy
Mean pooling vs CLS pooling affects:
- Quality
- Stability
- Memory footprint

---

## 8. Embeddings in RAG Systems

Typical RAG pipeline:

1. Chunk documents
2. Generate embeddings
3. Store in vector database
4. Retrieve top-k relevant chunks
5. Send to LLM for generation

Embedding quality directly affects RAG accuracy.

---

## 9. Choosing the Right Embedding Model

### Choose SMALL when:
- Millions of documents
- Low latency required
- Limited GPU/CPU
- Cost-sensitive deployment

### Choose LARGE when:
- High precision matters
- Small-to-medium corpus
- Complex semantic queries
- Legal, medical, or research use cases

---

## 10. Vector Database Considerations

Embedding dimension affects:
- Index size
- RAM usage
- Query speed

Rule of thumb:
- 384-dim → scalable
- 768-dim → balanced
- 1024-dim → premium quality

---

## 11. Best Practices

- Normalize embeddings
- Cap max token length
- Use FP16/BF16
- Benchmark with real queries
- Don’t over-optimize dimension unnecessarily

---

## 12. Key Takeaway

> Embedding models are **semantic compressors**.
> They trade text generation for **fast, scalable understanding**.

They are a foundational building block for modern search, RAG, and AI systems.
