"""
╔══════════════════════════════════════════════════════════════╗
║         DEMO 1 — Pure RAG (No Frameworks)                    ║
║         Email Summarizer — Local Knowledge Base              ║
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
  - No live emails — uses only static sample emails
  - Verbose code — we're doing everything manually

NOTE on embeddings:
  sentence-transformers runs LOCALLY — no API key needed, no cost.
  The model (~90MB) is downloaded once and cached on your machine.

NOTE on chat:
  Groq offers a free API tier (no credit card needed).
  Get your key at: https://console.groq.com
  Model: openai/gpt-oss-20b — fast, free-tier friendly, and current.

  UPDATE (Sep 2026): llama-3.1-8b-instant and llama-3.3-70b-versatile
  were deprecated and fully decommissioned by Groq on 16 Aug 2026.
  Official 1:1 replacements:
    llama-3.1-8b-instant   -> openai/gpt-oss-20b
    llama-3.3-70b-versatile -> openai/gpt-oss-120b
  Check https://console.groq.com/docs/deprecations if this breaks again.

NOTE on FAISS persistence:
  The FAISS index + chunks are saved to ./faiss_store/ on first run.
  Subsequent runs skip embedding entirely and load from disk.
  To force a fresh re-embed, delete the ./faiss_store/ folder.
"""

import os
import pickle
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
EMBED_MODEL   = "all-MiniLM-L6-v2"        # local, free, fast (~90MB download once)

# Primary chat model — override via GROQ_MODEL env var if you want to pin one.
CHAT_MODEL    = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Fallback chain tried in order if the primary model 404s / is deprecated /
# account has no access. Keeps the app alive across Groq's model churn.
FALLBACK_MODELS = [
    model for model in [
        CHAT_MODEL,
        "openai/gpt-oss-20b",    # replaces llama-3.1-8b-instant (fast, free-tier)
        "openai/gpt-oss-120b",   # replaces llama-3.3-70b-versatile (bigger, smarter)
    ] if model
]
# de-dupe while preserving order (in case CHAT_MODEL == one of the fallbacks)
FALLBACK_MODELS = list(dict.fromkeys(FALLBACK_MODELS))

CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 50     # overlap to avoid cutting context mid-sentence
TOP_K         = 4      # how many chunks to retrieve

# Persistence — FAISS index + chunks saved after first embed
FAISS_STORE_DIR   = "./faiss_store"
FAISS_INDEX_PATH  = os.path.join(FAISS_STORE_DIR, "index.faiss")
FAISS_CHUNKS_PATH = os.path.join(FAISS_STORE_DIR, "chunks.pkl")

# Clients
groq_client = Groq(api_key=GROQ_API_KEY)


def call_groq_with_fallback(messages: list[dict]) -> object:
    """Try the configured Groq model, then known-good fallbacks in order.

    Only falls through to the next model on a "model not found / no access"
    style error. Any other error (auth, rate limit, bad request) is raised
    immediately so it doesn't get masked by a retry loop.
    """
    last_error = None
    for i, model_name in enumerate(FALLBACK_MODELS):
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,          # gpt-oss models spend part of this on reasoning
                reasoning_effort="low",   # gpt-oss-only param; keeps more budget for the answer
            )
            if i > 0:
                print(f"   ⚠️  Primary model unavailable — fell back to: {model_name}")
            content = response.choices[0].message.content
            if not content or not content.strip():
                # Reasoning ate the whole budget — retry once with a bigger cap, no reasoning cap trick needed
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
                    reasoning_effort="low",
                )
            return response
        except Exception as exc:  # pragma: no cover - depends on Groq availability
            last_error = exc
            error_text = str(exc).lower()
            is_model_issue = (
                "model" in error_text
                or "not found" in error_text
                or "access" in error_text
                or "decommission" in error_text
            )
            if not is_model_issue:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Groq model available for this account.")


# Load embedding model once at startup (downloads ~90MB on very first run)
print("⏳ Loading local embedding model (downloaded once, then cached)...")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"✅ Embedding model '{EMBED_MODEL}' ready\n")


# ── Step 1: Load sample emails ────────────────────────────────────────────────
def load_sample_emails() -> list[dict]:
    """
    Returns a list of {source, content} dicts representing sample emails.

    In Demo 1 we use hardcoded samples to focus on the RAG mechanics.
    Demo 3 will replace this with LIVE Gmail data via MCP.
    """
    emails = [
        {
            "source": "email_001",
            "content": (
                "From: priya.sharma@company.com\n"
                "To: team@company.com\n"
                "Subject: Q3 Budget Review — Action Required\n"
                "Date: Mon, 14 Oct 2024 09:15:00\n\n"
                "Hi team,\n\n"
                "Please review the attached Q3 budget spreadsheet by Wednesday EOD. "
                "We are 12% over on cloud infrastructure and need to identify cuts. "
                "Each team lead must submit a one-paragraph justification for any line item "
                "above ₹2 lakh. Finance sign-off is needed before the board meeting on Friday.\n\n"
                "Key action items:\n"
                "1. Review your team's line items\n"
                "2. Submit justifications by Wed 5 PM\n"
                "3. Attend budget sync Thursday 3 PM\n\n"
                "Regards, Priya"
            )
        },
        {
            "source": "email_002",
            "content": (
                "From: recruiter@techcorp.io\n"
                "To: you@company.com\n"
                "Subject: Interview Invitation — Senior ML Engineer\n"
                "Date: Tue, 15 Oct 2024 11:30:00\n\n"
                "Dear Candidate,\n\n"
                "We were impressed by your profile and would like to invite you for a "
                "technical interview for the Senior ML Engineer position at TechCorp.\n\n"
                "Round 1: System design (45 min) — Thursday 16 Oct, 2 PM IST\n"
                "Round 2: Coding (1 hr) — Friday 17 Oct, 11 AM IST\n\n"
                "Please confirm availability by replying to this email. "
                "A Google Meet link will be shared upon confirmation.\n\n"
                "Best, Neha (Talent Acquisition, TechCorp)"
            )
        },
        {
            "source": "email_003",
            "content": (
                "From: alerts@aws.amazon.com\n"
                "To: devops@company.com\n"
                "Subject: [URGENT] Cost Anomaly Detected — $4,200 spike\n"
                "Date: Wed, 16 Oct 2024 03:45:00\n\n"
                "AWS Cost Anomaly Detection has identified an unusual spend pattern.\n\n"
                "Service: Amazon EC2\n"
                "Expected daily cost: $180\n"
                "Actual cost (last 24h): $4,380\n"
                "Anomaly started: 16 Oct 2024 00:00 UTC\n\n"
                "Likely cause: Auto Scaling group launched 22 additional instances in ap-south-1. "
                "Recommended action: Review ASG configuration and set a max capacity limit. "
                "Check CloudWatch for the triggering alarm.\n\n"
                "View in AWS Console: https://console.aws.amazon.com/cost-management/anomalies"
            )
        },
        {
            "source": "email_004",
            "content": (
                "From: rajesh.nair@vendor.com\n"
                "To: procurement@company.com\n"
                "Subject: Contract Renewal — SaaS Subscription (Invoice #INV-2024-089)\n"
                "Date: Wed, 16 Oct 2024 10:00:00\n\n"
                "Dear Team,\n\n"
                "Your annual subscription for DataSync Pro expires on 31 Oct 2024. "
                "To avoid service interruption, please process renewal by 25 Oct.\n\n"
                "Renewal details:\n"
                "- Plan: Enterprise (500 seats)\n"
                "- Amount: ₹18,00,000 + GST\n"
                "- Term: 12 months (Nov 2024 – Oct 2025)\n\n"
                "Early renewal discount of 8% available if payment received by 20 Oct. "
                "Please reach out to discuss any changes to seat count or plan tier.\n\n"
                "Regards, Rajesh (Account Manager, DataSync)"
            )
        },
        {
            "source": "email_005",
            "content": (
                "From: cto@company.com\n"
                "To: engineering@company.com\n"
                "Subject: All-hands Engineering Sync — Agenda\n"
                "Date: Thu, 17 Oct 2024 08:00:00\n\n"
                "Team,\n\n"
                "Our quarterly all-hands is this Friday 18 Oct at 10 AM in the main conference room (and Zoom).\n\n"
                "Agenda:\n"
                "1. H2 roadmap review — 30 min\n"
                "2. AI/ML platform update — 15 min\n"
                "3. Hiring plan (10 new roles) — 10 min\n"
                "4. Q&A — 15 min\n\n"
                "Please come prepared with blockers or items you'd like raised. "
                "The Zoom link is in the calendar invite. Recordings will be shared post-session.\n\n"
                "See you Friday, Vikram (CTO)"
            )
        },
        {
            "source": "email_006",
            "content": (
                "From: noreply@github.com\n"
                "To: dev@company.com\n"
                "Subject: [GitHub] Security alert for dependency lodash in repo api-service\n"
                "Date: Thu, 17 Oct 2024 14:22:00\n\n"
                "A high-severity security vulnerability has been found in lodash < 4.17.21 "
                "used by your repository api-service.\n\n"
                "CVE: CVE-2021-23337\n"
                "Severity: HIGH\n"
                "Affected version: 4.17.15\n"
                "Fixed version: 4.17.21\n\n"
                "Recommended action: Update lodash to >= 4.17.21 in package.json and run npm audit fix. "
                "Dependabot has opened pull request #142 with the fix. "
                "Please review and merge at the earliest.\n\n"
                "GitHub Security"
            )
        },
        {
            "source": "email_007",
            "content": (
                "From: hr@company.com\n"
                "To: all@company.com\n"
                "Subject: Diwali Bonus & Holiday Schedule\n"
                "Date: Fri, 18 Oct 2024 09:00:00\n\n"
                "Dear All,\n\n"
                "We are pleased to announce the Diwali bonus will be credited to your accounts by 28 October. "
                "The amount will be equivalent to one month's basic salary.\n\n"
                "Holiday schedule:\n"
                "- Diwali holiday: 1 Nov (Friday)\n"
                "- Office resumes: 4 Nov (Monday)\n\n"
                "For those working remotely, please ensure your manager approves the leave in the portal. "
                "Wishing everyone a happy and safe Diwali!\n\n"
                "HR Team"
            )
        },
        {
            "source": "email_008",
            "content": (
                "From: support@stripe.com\n"
                "To: billing@company.com\n"
                "Subject: Payment failed — update your billing details\n"
                "Date: Fri, 18 Oct 2024 16:05:00\n\n"
                "Your monthly payment of $499.00 for Stripe's standard plan failed on 18 Oct 2024.\n\n"
                "Reason: Card declined (insufficient funds)\n"
                "Next retry: 21 Oct 2024\n\n"
                "To avoid service interruption, please update your payment method at "
                "https://dashboard.stripe.com/settings/billing before the retry date. "
                "If payment is not received within 7 days, your account may be restricted.\n\n"
                "Stripe Billing Support"
            )
        },
    ]

    print(f"📧 Loaded {len(emails)} sample emails from local store\n")
    return emails


# ── Step 2: Chunk emails ───────────────────────────────────────────────────────
def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Splits each email into overlapping chunks.

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

    Sends retrieved email context + user question to Groq (openai/gpt-oss-20b).
    Groq's free tier is very fast and needs no credit card.
    """
    context = "\n\n---\n\n".join(
        f"[Email {i}: {c['source']}]\n{c['content']}"
        for i, c in enumerate(retrieved_chunks, 1)
    )

    try:
        response = call_groq_with_fallback([
            {
                "role": "system",
                "content": (
                    "You are a helpful email assistant. "
                    "Summarize and answer questions about emails using ONLY the context provided. "
                    "Be concise. Highlight key action items, deadlines, and important figures. "
                    "If the answer isn't in the context, say: "
                    "'I don't have that email in my current index.'"
                )
            },
            {
                "role": "user",
                "content": f"EMAIL CONTEXT:\n{context}\n\nQUESTION: {query}"
            }
        ])
        content = response.choices[0].message.content
        if not content or not content.strip():
            return "⚠️ Model returned an empty response (likely used its full token budget on reasoning). Try increasing max_tokens further, or ask a more specific question."
        return content.strip()

    except Exception as e:
        return f"⚠️ Groq API error: {e}"


# ── Main Chat Loop ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  📧 Email Summarizer — Demo 1: Pure RAG")
    print("  📌 Embeddings: local (sentence-transformers)")
    print(f"  📌 Chat:       Groq API  ({CHAT_MODEL})")
    print("=" * 60)
    print()

    # Load from disk if already built; otherwise build and save
    cached = load_vector_store()
    if cached:
        index, chunks = cached
    else:
        emails        = load_sample_emails()
        chunks        = chunk_documents(emails)
        index, chunks = build_vector_store(chunks)

    print("✅ Email index ready! Ask me anything about your emails.\n")
    print("   Try asking:")
    print("   - Summarize all urgent emails")
    print("   - What emails need my action today?")
    print("   - Is there anything about AWS costs?")
    print("   - What is the Diwali bonus amount?\n")
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
        # Try: "Show me emails from today in my real Gmail" → no live data → Demo 3


if __name__ == "__main__":
    main()
