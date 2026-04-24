"""
╔══════════════════════════════════════════════════════════════╗
║         DEMO 1 — Pure RAG (No Frameworks)                    ║
║         GitHub Repo Assistant — FastAPI Docs                 ║
║                                                              ║
║  Stack:  sentence-transformers (local) + FAISS + Groq        ║
║  Flow:   Ingest → Chunk → Embed → Store → Retrieve → Answer ║
╚══════════════════════════════════════════════════════════════╝

What this demo teaches:
  - What RAG is and why we need it
  - How documents become vectors
  - How similarity search works
  - The raw prompt augmentation pattern

What breaks at the end (motivates Demo 2):
  - No memory — follow-up questions lose context
  - No live data — can't fetch GitHub issues/PRs
  - Verbose code — we're doing everything manually

NOTE on embeddings:
  sentence-transformers runs LOCALLY — no API key needed, no cost.
  The model (~90MB) is downloaded once and cached on your machine.

NOTE on chat:
  Groq offers a free API tier (no credit card needed).
  Get your key at: https://console.groq.com
  Model: llama-3.1-8b-instant — fast, free, and reliable.

NOTE on FAISS persistence:
  The FAISS index + chunks are saved to ./faiss_store/ on first run.
  Subsequent runs skip fetching + embedding entirely and load from disk.
  To force a fresh re-embed, delete the ./faiss_store/ folder.
"""

import os
import pickle
from dotenv import load_dotenv
load_dotenv()

import requests
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
EMBED_MODEL   = "all-MiniLM-L6-v2"        # local, free, fast (~90MB download once)
CHAT_MODEL    = "llama-3.1-8b-instant"    # Groq free tier

CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 50     # overlap to avoid cutting context mid-sentence
TOP_K         = 4      # how many chunks to retrieve

# Persistence — FAISS index + chunks saved after first embed
FAISS_STORE_DIR   = "./faiss_store"
FAISS_INDEX_PATH  = os.path.join(FAISS_STORE_DIR, "index.faiss")
FAISS_CHUNKS_PATH = os.path.join(FAISS_STORE_DIR, "chunks.pkl")

# Clients
groq_client = Groq(api_key=GROQ_API_KEY)

# Load embedding model once at startup (downloads ~90MB on very first run)
print("⏳ Loading local embedding model (downloaded once, then cached)...")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"✅ Embedding model '{EMBED_MODEL}' ready\n")


# ── Step 1: Fetch raw content from GitHub ────────────────────────────────────
def fetch_github_content() -> list[dict]:
    """
    Fetches README and docs files from the FastAPI repo.
    Returns a list of {source, content} dicts.
    """
    urls = {
        "README":              "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
        "CONTRIBUTING":        "https://raw.githubusercontent.com/tiangolo/fastapi/master/CONTRIBUTING.md",
        "docs/features":       "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/features.md",
        "docs/tutorial_intro": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/index.md",
        "docs/first_steps":    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/first-steps.md",
        "docs/path_params":    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-params.md",
        "docs/query_params":   "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/query-params.md",
        "docs/body":           "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/body.md",
        "docs/deployment":     "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/deployment/index.md",
    }

    documents = []
    print("📥 Fetching documents from GitHub...")
    for name, url in urls.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                documents.append({"source": name, "content": response.text})
                print(f"   ✅ {name} ({len(response.text)} chars)")
            else:
                print(f"   ⚠️  Skipped {name} (HTTP {response.status_code})")
        except Exception as e:
            print(f"   ❌ Failed {name}: {e}")

    print(f"\n📄 Total documents fetched: {len(documents)}\n")
    return documents


# ── Step 2: Chunk documents ───────────────────────────────────────────────────
def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Splits each document into overlapping chunks.

    Why chunk?
      - LLMs have context limits
      - Smaller chunks = more precise retrieval
      - Overlap prevents losing context at boundaries
    """
    chunks = []
    for doc in documents:
        text   = doc["content"]
        source = doc["source"]
        start  = 0
        chunk_index = 0
        while start < len(text):
            chunk_text = text[start : start + CHUNK_SIZE]
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "source":      source,
                    "chunk_index": chunk_index,
                    "content":     chunk_text.strip()
                })
                chunk_index += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"✂️  Total chunks created: {len(chunks)}")
    print(f"   Average chunk size: {np.mean([len(c['content']) for c in chunks]):.0f} chars\n")
    return chunks


# ── Step 3: Generate embeddings (LOCAL — zero API cost) ───────────────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Converts text into numerical vectors using a local sentence-transformers model.

    Why local embeddings?
      - Zero API cost — runs on your CPU
      - No rate limits or quota to worry about
      - all-MiniLM-L6-v2 is fast and very good for retrieval tasks
    """
    vectors = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype("float32")


# ── Step 4: Build vector store and persist to disk ────────────────────────────
def build_vector_store(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    Embeds all chunks, stores them in a FAISS index, then saves both to disk.

    FAISS = Facebook AI Similarity Search
      - In-memory vector database, extremely fast nearest-neighbour search
      - Persisted to ./faiss_store/ so next startup skips this entire step
    """
    print("🔢 Generating embeddings locally (no API calls)...")
    texts = [c["content"] for c in chunks]

    batch_size  = 100
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        all_vectors.append(embed_texts(texts[i : i + batch_size]))
        print(f"   Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

    all_vectors = np.vstack(all_vectors)
    dimension   = all_vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(all_vectors)

    os.makedirs(FAISS_STORE_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n📦 Vector store built and saved to {FAISS_STORE_DIR}/")
    print(f"   Vectors stored: {index.ntotal}  |  Dimensions: {dimension}\n")
    return index, chunks


def load_vector_store() -> tuple[faiss.Index, list[dict]] | None:
    """Loads previously saved FAISS index + chunks. Returns None if not found."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_CHUNKS_PATH):
        print(f"📦 Loading FAISS store from disk ({FAISS_STORE_DIR}/)...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"   ✅ {index.ntotal} vectors, {len(chunks)} chunks — skipping embed step\n")
        return index, chunks
    return None


# ── Step 5: Retrieve relevant chunks ─────────────────────────────────────────
def retrieve(query: str, index: faiss.Index, chunks: list[dict]) -> list[dict]:
    """
    Embed the query and find the TOP_K most similar chunks via FAISS.

    How it works:
      1. Embed the query into the same vector space as the chunks
      2. Compute distances between query vector and all chunk vectors
      3. Return the k nearest neighbours
    """
    query_vector        = embed_texts([query])
    distances, indices  = index.search(query_vector, TOP_K)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["distance"] = float(dist)
            results.append(chunk)
    return results


# ── Step 6: Generate answer via Groq ─────────────────────────────────────────
def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    The 'Augmented Generation' part of RAG.

    Sends retrieved context + user question to Groq (llama-3.1-8b-instant).
    Groq's free tier is very fast (~500 tok/s) and needs no credit card.
    """
    context = "\n\n---\n\n".join(
        f"[Source {i}: {c['source']}]\n{c['content']}"
        for i, c in enumerate(retrieved_chunks, 1)
    )

    try:
        response = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for the FastAPI GitHub repository. "
                        "Answer the user's question using ONLY the context provided. "
                        "If the answer isn't in the context, say: "
                        "'I don't have enough information about that in the docs I've indexed.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"
                }
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Groq API error: {e}"


# ── Main Chat Loop ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🤖 GitHub Repo Assistant — Demo 1: Pure RAG")
    print("  📌 Embeddings: local (sentence-transformers)")
    print("  📌 Chat:       Groq API  (llama-3.1-8b-instant)")
    print("=" * 60)
    print()

    # Load from disk if already built; otherwise build and save
    cached = load_vector_store()
    if cached:
        index, chunks = cached
    else:
        documents     = fetch_github_content()
        chunks        = chunk_documents(documents)
        index, chunks = build_vector_store(chunks)

    print("✅ Knowledge base ready! Ask me anything about FastAPI.\n")
    print("   Try asking:")
    print("   - What is FastAPI?")
    print("   - How do I define a path parameter?")
    print("   - How do I deploy FastAPI?")
    print("   - What are the main features of FastAPI?\n")
    print("   [Type 'quit' to exit]\n")
    print("-" * 60)

    while True:
        query = input("\n🧑 You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        retrieved = retrieve(query, index, chunks)
        print(f"\n📚 Retrieved {len(retrieved)} chunks:")
        for i, chunk in enumerate(retrieved, 1):
            print(f"   {i}. [{chunk['source']}] dist={chunk['distance']:.3f} — \"{chunk['content'][:60]}...\"")

        print("\n🤖 Assistant: ", end="", flush=True)
        print(generate_answer(query, retrieved))

        # ── THE CLIFFHANGER ──
        # Try: "What did I just ask you?" → won't remember (no memory) → Demo 2
        # Try: "What are the open issues about performance?" → no live data → Demo 3


if __name__ == "__main__":
    main()
