# 🤖 GitHub Repo Assistant — Workshop Demos

Three progressive demos showing RAG → LangChain → MCP,
all built around answering questions about the FastAPI repository.

**Model stack (fully free):**
| Role | Model | Where it runs | Cost |
|------|-------|---------------|------|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Locally on your CPU | Free |
| Chat / Agent | `llama-3.1-8b-instant` | Groq Inference API | Free tier |

---

## Setup (do this once before any demo)

```bash
# 1. Clone / navigate to this folder
cd github-repo-assistant

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up your API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
# Optionally add GITHUB_TOKEN to avoid GitHub rate limits in Demo 3
```

### Getting a free Groq API key
1. Go to [console.groq.com](https://console.groq.com) and sign up (no credit card needed)
2. Click **API Keys → Create API Key**
3. Copy the key (starts with `gsk_`) into your `.env` file as `GROQ_API_KEY`

### First run note
On the very first run, `sentence-transformers` downloads the embedding model
(`all-MiniLM-L6-v2`, ~90MB) and caches it locally. Subsequent runs are instant.

---

## Demo 1 — Pure RAG

**Teaches:** Chunking, local embeddings, FAISS, similarity search, prompt augmentation

```bash
python rag_assistant.py
```

**FAISS persistence:** The index is saved to `./faiss_store/` after the first run.
Subsequent runs load from disk instantly — no re-fetching or re-embedding.
Delete `./faiss_store/` to force a rebuild.

**Cliffhanger questions at the end:**
> "What did I just ask you?" → Will fail. No memory. Sets up Demo 2.
> "What are the open GitHub issues about performance?" → Will fail. No live data. Sets up Demo 3.

---

## Demo 2 — RAG + LangChain

**Teaches:** LangChain chains, RecursiveTextSplitter, Chroma persistence, ConversationMemory

```bash
python rag_langchain_assistant.py
```

**Note:** On first run it embeds docs and saves to `./chroma_db/`.
Subsequent runs load from disk instantly. Delete `./chroma_db/` to force rebuild.

**Show the audience:**
- Ask a follow-up like "Can you show me an example?" after any answer
- Ask "What did I first ask you?" — memory works now!

**Cliffhanger question at the end:**
> "What are the open GitHub issues about WebSockets?"
> → Will fail. We have no live tools. Sets up Demo 3.

---

## Demo 3 — RAG + LangChain + MCP

**Teaches:** Agent reasoning (function calling), tool selection, MCP pattern, live GitHub data

```bash
python rag_mcp_assistant.py
```

**Note:** Reuses `./chroma_db/` if it exists from Demo 2. Otherwise builds fresh.
`verbose=True` by default so the audience sees the agent's tool selection.

Because Groq's Llama supports OpenAI-compatible function calling, Demo 3 uses
`create_openai_functions_agent` — more reliable tool selection than ReAct text loops.

**Key demo moment — ask this sequence:**
1. "How do I define a path parameter?" → uses RAG tool
2. "Are there any open issues about path params?" → uses GitHub issues tool
3. "How many stars does FastAPI have?" → uses repo stats tool
4. "Compare what the docs say about routing vs any open routing issues" → MULTIPLE tools

---

## Architecture at a Glance

```
Demo 1: User → [Embed Query (local)] → FAISS → [Top-K chunks] → [Prompt] → Groq → Answer

Demo 2: User → LangChain Chain → Chroma Retriever → [Augmented Prompt + Memory] → Groq → Answer

Demo 3: User → LangChain Agent → [Reads tool descriptions] → Picks tool(s):
                                    ├── search_fastapi_docs    (Chroma RAG)
                                    ├── search_github_issues   (Live GitHub API)
                                    ├── search_github_prs      (Live GitHub API)
                                    ├── browse_repo_files      (Live GitHub API)
                                    └── get_repo_stats         (Live GitHub API)
                                  → Synthesizes results → Answer
```

---

## What MCP Means in Demo 3

In Demo 3, the agent discovers tools by reading their **descriptions** — it's never
hardcoded to call a specific tool for a specific query. This is the MCP pattern:

- Tools **self-describe** what they do
- Agents **reason** about which tool fits the query (via function calling)
- New tools can be added without changing agent logic

In production, these tool descriptions would live on an **MCP server**, and the agent
would auto-discover them at runtime.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `faiss` install fails | Try `pip install faiss-cpu --no-cache-dir` |
| `sentence-transformers` slow first run | Normal — downloading ~90MB model once |
| Groq rate limit hit | Free tier is generous; wait a minute and retry |
| GitHub rate limit errors | Add `GITHUB_TOKEN` to your `.env` |
| Chroma errors on load | Delete `chroma_db/` folder and let it rebuild |
| FAISS errors on load (Demo 1) | Delete `faiss_store/` folder and let it rebuild |
| `hub.pull` fails (Demo 3) | Run `pip install langchainhub` and check internet access |
