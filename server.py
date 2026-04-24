"""
╔══════════════════════════════════════════════════════════════╗
║  Web Server — wraps Demo 1 / 2 / 3 behind a REST + SSE API  ║
║                                                              ║
║  To switch demos, change ACTIVE_DEMO at the top.            ║
║  Then run:  python server.py                                 ║
║  Frontend:  open index.html in your browser                 ║
╚══════════════════════════════════════════════════════════════╝

Install extra deps (in addition to your existing requirements):
  pip install fastapi uvicorn sse-starlette

Run:
  python server.py
  → http://localhost:8000
"""

import os
import sys
import asyncio
import json
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

# ─────────────────────────────────────────────────────────────
#  👇 CHANGE THIS to switch between demos: 1, 2, or 3
# ─────────────────────────────────────────────────────────────
ACTIVE_DEMO = 2
# ─────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Bootstrap the selected demo ───────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  🚀 Starting Demo {ACTIVE_DEMO}")
print(f"{'='*60}\n")

if ACTIVE_DEMO == 1:
    # ── Demo 1: Pure RAG (FAISS) ──────────────────────────────────────────────
    from rag_assistant import (
        load_vector_store, fetch_github_content,
        chunk_documents, build_vector_store,
        retrieve, generate_answer
    )

    cached = load_vector_store()
    if cached:
        index, chunks = cached
    else:
        documents     = fetch_github_content()
        chunks_raw    = chunk_documents(documents)
        index, chunks = build_vector_store(chunks_raw)

    DEMO_LABEL = "Demo 1 — Pure RAG (FAISS + sentence-transformers + Groq)"
    DEMO_TIPS  = [
        "What is FastAPI?",
        "How do I define a path parameter?",
        "How do I deploy FastAPI?",
        "What are the main features of FastAPI?",
        "What did I just ask you? (will fail — no memory)",
    ]

    def get_answer_with_log(query: str):
        """Returns (answer_text, log_lines)"""
        retrieved  = retrieve(query, index, chunks)
        log_lines  = [f"📚 Retrieved {len(retrieved)} chunks:"]
        for i, chunk in enumerate(retrieved, 1):
            log_lines.append(
                f"   {i}. [{chunk['source']}] dist={chunk['distance']:.3f} "
                f"— \"{chunk['content'][:60]}...\""
            )
        answer = generate_answer(query, retrieved)
        return answer, log_lines


elif ACTIVE_DEMO == 2:
    # ── Demo 2: RAG + LangChain (Chroma + Memory) ─────────────────────────────
    from rag_langchain_assistant import (
        load_documents, split_documents, get_vector_store, build_chain
    )

    documents   = load_documents()
    chunks      = split_documents(documents)
    vectorstore = get_vector_store(chunks)
    chain       = build_chain(vectorstore)

    DEMO_LABEL = "Demo 2 — RAG + LangChain (Chroma + ConversationMemory + Groq)"
    DEMO_TIPS  = [
        "What are path parameters in FastAPI?",
        "Can you show me an example? (tests memory)",
        "What did I first ask you? (tests memory)",
        "What are the open GitHub issues? (will fail — no live data)",
    ]

    def get_answer_with_log(query: str):
        result  = chain.invoke({"question": query})
        sources = result.get("source_documents", [])
        log_lines = []
        if sources:
            srcs = ", ".join(set(s.metadata["source"] for s in sources))
            log_lines.append(f"📚 Sources used: {srcs}")
        return result["answer"], log_lines


elif ACTIVE_DEMO == 3:
    # ── Demo 3: RAG + LangChain + MCP (Agent + Live GitHub) ──────────────────
    from rag_mcp_assistant import get_or_build_vectorstore, build_agent

    vectorstore = get_or_build_vectorstore()
    agent       = build_agent(vectorstore)

    DEMO_LABEL = "Demo 3 — RAG + LangChain + MCP (Agent + Live GitHub + Groq)"
    DEMO_TIPS  = [
        "How do I define a path parameter? (→ RAG tool)",
        "Are there any open issues about WebSockets? (→ GitHub issues)",
        "What features are being worked on right now? (→ GitHub PRs)",
        "How many stars does FastAPI have? (→ repo stats)",
        "Compare what the docs say about routing vs open routing issues (→ multi-tool)",
    ]

    def get_answer_with_log(query: str):
        result = agent.invoke({"input": query})
        log_lines = ["🔧 Agent selected tools automatically (see server terminal for reasoning)"]
        return result["output"], log_lines

else:
    raise ValueError(f"ACTIVE_DEMO must be 1, 2, or 3 — got {ACTIVE_DEMO}")


print(f"\n✅ {DEMO_LABEL} ready\n")


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/info")
def get_info():
    return {
        "demo":  ACTIVE_DEMO,
        "label": DEMO_LABEL,
        "tips":  DEMO_TIPS,
    }


@app.post("/chat")
async def chat(request: Request):
    """
    SSE endpoint — streams log lines then the final answer.
    Client receives events:
      { type: "log",    data: "..." }
      { type: "answer", data: "..." }
      { type: "done"               }
      { type: "error",  data: "..." }
    """
    body  = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "empty query"}, status_code=400)

    async def event_stream():
        try:
            # Run synchronous LangChain/FAISS code in thread pool
            loop   = asyncio.get_event_loop()
            answer, log_lines = await loop.run_in_executor(
                None, get_answer_with_log, query
            )

            for line in log_lines:
                yield {"event": "log", "data": json.dumps(line)}
                await asyncio.sleep(0)   # yield control

            yield {"event": "answer", "data": json.dumps(answer)}
            yield {"event": "done",   "data": ""}

        except Exception as e:
            yield {"event": "error", "data": json.dumps(str(e))}

    return EventSourceResponse(event_stream())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")