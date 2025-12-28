
# Embedding Models — Principles, Theory, and Practice
*A deep yet practical guide, with BGE models as concrete examples*

---

## 1. What Is an Embedding Model?

An **embedding model** maps variable-length input data (usually text) into a **fixed-length numerical vector** that captures **semantic meaning**.

Key properties:
- Similar meanings → vectors close together
- Different meanings → vectors far apart
- Output size is **fixed**, regardless of input length

Embedding models are **not generative**.  
They are optimized for *understanding, matching, and retrieval*.

---

## 2. Embedding Models vs Large Language Models (LLMs)

| Aspect | Embedding Model | LLM |
|------|----------------|-----|
| Output | Vector | Text |
| Input length | Limited | Limited |
| Objective | Semantic similarity | Next-token prediction |
| Information retained | Meaning | Meaning + form |
| Example | BGE, E5 | GPT, LLaMA |
| Typical role | Retriever | Generator |

Embedding models are usually **paired with LLMs** in RAG systems.

---

## 3. Example: BAAI BGE Models

### bge-small-zh
- 384-dimensional embeddings
- Lightweight, fast
- Suitable for large-scale retrieval

### bge-large-zh-v1.5
- 1024-dimensional embeddings
- Higher semantic precision
- Higher memory and compute cost

Embedding dimension controls the **capacity of semantic representation**.

---

## 4. How Embeddings Are Computed

### 4.1 Tokenization
Input text is split into **tokens**.

Example:
```
Text → [token₁, token₂, ..., tokenₙ]
```

### 4.2 Token-level Representations
Each token is mapped to a vector of size `hidden_dim`.

```
tokenᵢ → hᵢ ∈ ℝᴰ
```

### 4.3 Pooling (Sentence-Level Embedding)

Since outputs are token-level, pooling is required to produce **one vector per input**.

#### Mean Pooling (recommended)
```
e = (1/N) Σ hᵢ   (ignoring padding tokens)
```

- Uses all tokens
- Stable and robust
- Best aligned with contrastive training

#### CLS Token Pooling
- Uses the special `[CLS]` token
- Works only if model was trained for it

#### Max Pooling
- Takes max value per dimension
- Highlights strong signals but is noisy

| Pooling | Stability | Semantic Quality |
|-------|----------|------------------|
| Mean | High | High |
| CLS | Medium | Medium |
| Max | Low | Low |

---

## 5. Embedding Models as Information Bottlenecks

### 5.1 Information Bottleneck Theory (IB)

IB theory asks:

> How do we compress input data while preserving only task-relevant information?

It defines:
- **X**: input (tokens)
- **Y**: task (semantic similarity / relevance)
- **Z**: compressed representation (embedding)

Goal:
```
min I(X; Z)  while  max I(Z; Y)
```

Meaning:
- Forget irrelevant details
- Preserve meaning needed for the task

---

### 5.2 Why Embeddings Are Semantic Compressors

Embeddings:
- Discard syntax, order, and surface form
- Preserve intent, topic, and meaning
- Are **lossy by design**

They are not byte compressors — they are **meaning compressors**.

---

## 6. Token Budget and Sequence Length

- Models can only read up to `max_length` tokens
- Tokens beyond this limit are **discarded**
- Lost information cannot be recovered

This makes embeddings **strictly capacity-limited**.

---

## 7. Chunking Long Documents

### Why chunking is required
Long text > token limit → truncation → meaning loss

### Why chunk by lines / paragraphs
- Language-agnostic
- Robust to lists, code, markdown
- Preserves document structure

### Best practice
1. Split by paragraph or line
2. Merge until token limit
3. Add overlap (10–20%)

Chunking decides **what meaning survives the bottleneck**.

---

## 8. Embedding Dimension = Bottleneck Width

| Dimension | Capacity | Trade-off |
|--------|---------|----------|
| 384 | Narrow | Fast, scalable |
| 768 | Medium | Balanced |
| 1024 | Wide | High precision |

Higher dimension:
- Retains more information
- Costs more memory and compute

---

## 9. Similarity Metrics

Common choices:
- **Cosine similarity** (most common)
- Dot product
- Euclidean distance

Cosine similarity compares **direction**, not magnitude.

---

## 10. Precision Formats

### FP16
- Floating Point 16-bit
- Higher precision, smaller range

### BF16
- Brain Floating Point 16-bit
- Larger range, more stable

Both:
- Use 16 bits
- Reduce VRAM vs FP32
- Suitable for inference

---

## 11. VRAM Usage Explained

Approximation:
```
VRAM ≈ model weights + activations × batch_size × sequence_length²
```

### Model weights
- Fixed cost
- Depends on parameter count and precision

### Activations
- Intermediate tensors
- Dominated by attention layers
- Scale quadratically with sequence length

This is why:
- Truncation matters
- Batch size matters
- Small models can use more VRAM if misconfigured

---

## 12. Frameworks and Runtimes

### PyTorch
- Training and inference
- Flexible, research-friendly
- Higher overhead

### ONNX / ONNX Runtime
- Framework-agnostic inference
- Faster and more stable

### TensorRT
- NVIDIA-optimized inference
- Fastest GPU performance
- Production-focused

---

## 13. Embeddings in RAG Pipelines

Typical flow:
1. Chunk documents
2. Embed chunks
3. Store vectors
4. Retrieve top-k
5. Feed to LLM

Embedding quality directly determines RAG quality.

---

## 14. Best Practices

- Use mean pooling
- Normalize embeddings
- Cap sequence length (256–512)
- Use FP16/BF16
- Benchmark with real queries

---

## 15. Final Mental Model

> An embedding is a **semantic fingerprint** produced by a **strict information bottleneck**.

It:
- Is fixed-size
- Is cheap to compare
- Loses detail
- Preserves meaning (as trained)

Understanding embeddings means understanding **what information is allowed through the bottleneck — and what is not**.


---

## 16. Attention Mechanism in Embedding Models

### 16.1 Is Attention Used?
Yes. Modern embedding models (BGE, E5, SBERT, etc.) are **Transformer encoder models** and rely heavily on **self-attention**.

They do **not** generate text, but they **do** use the same attention mechanism as models like BERT.

---

### 16.2 What Kind of Attention?
Embedding models use:

- **Self-attention**
- **Encoder-only architecture**
- **No decoder**
- **No cross-attention** (except in rerankers)

Each token attends to **all other tokens in the same sequence**.

---

### 16.3 Why Attention Is Essential for Embeddings

Attention enables:
- Context-aware token representations
- Disambiguation of word meaning
- Global understanding of a sequence

Example:
```
"bank" + "river"  ≠  "bank" + "money"
```

Without attention, embeddings would collapse into context-free word averages.

---

### 16.4 Attention and Semantic Compression

Attention and pooling play complementary roles:

- **Attention** → builds rich, contextual token meanings
- **Pooling** → compresses them into a fixed-size vector

Pipeline:
```
Tokens → Self-Attention → Contextual Token Vectors → Pooling → Embedding
```

This aligns with **Information Bottleneck Theory**:
- Attention extracts task-relevant signals
- Pooling enforces the bottleneck

---

### 16.5 Attention and VRAM Cost

Self-attention dominates memory usage.

Per layer complexity:
```
O(sequence_length² × hidden_dim)
```

Implications:
- Doubling sequence length ≈ 4× memory
- Long documents are expensive
- Chunking is mandatory for scalability

---

### 16.6 Embedding Models vs Older Non-Attention Models

| Model Type | Context-aware | Quality |
|----------|---------------|---------|
| Word2Vec / GloVe | ❌ No | Low |
| Transformer Embeddings | ✅ Yes | High |

Modern embedding quality is largely due to **attention**.

---

## 17. Final Unified View

> **Embedding models = Attention-based understanding + Bottlenecked semantic compression**

Attention gives embeddings meaning.  
The bottleneck makes them scalable.

Understanding both is key to designing high-quality retrieval and RAG systems.
