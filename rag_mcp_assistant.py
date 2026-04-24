"""
╔══════════════════════════════════════════════════════════════╗
║         DEMO 3 — RAG + LangChain + MCP                      ║
║         GitHub Repo Assistant — FastAPI Repo (LIVE)         ║
║                                                              ║
║  Stack:  LangChain Agent + Chroma + Groq + GitHub API       ║
║  Flow:   Query → Agent decides → RAG or Live Tool → Answer  ║
╚══════════════════════════════════════════════════════════════╝

What's new vs Demo 2:
  ✅ Agent replaces Chain    → reasons about WHICH tool to use
  ✅ RAG Tool                → same Chroma knowledge base, now a tool
  ✅ GitHub Live Tools       → fetches real issues, PRs, file trees
  ✅ MCP pattern             → tools are standardized, pluggable
  ✅ Memory still works      → nothing we built before is thrown away

NOTE on model choices:
  Embeddings → sentence-transformers (local, free)
  Agent LLM  → Groq API (llama-3.1-8b-instant, free tier)

  Unlike HuggingFace models, Groq's Llama supports OpenAI-compatible
  function calling — so we can use create_openai_functions_agent
  (same as the original OpenAI version), giving better tool selection.
"""

import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.tools import Tool
from langchain import hub

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN")   # optional — avoids rate limits
GITHUB_REPO        = "tiangolo/fastapi"
CHROMA_PERSIST_DIR = "./chroma_db"               # reuse from Demo 2 if it exists
COLLECTION_NAME    = "fastapi_docs"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, free
CHAT_MODEL  = "llama-3.1-8b-instant"                    # Groq free tier
TOP_K = 4


# ── Shared: GitHub API helper ─────────────────────────────────────────────────
def github_get(endpoint: str) -> dict | list:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    if endpoint:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/{endpoint}"
    else:
        url = f"https://api.github.com/repos/{GITHUB_REPO}"   # ✅ FIX

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS — the MCP pattern
#  Each tool has a name, description, and callable.
#  The agent reads descriptions and decides which tool(s) to call.
#  In a real MCP setup, these would be served by an MCP server and
#  auto-discovered at runtime — no code changes to add new tools.
# ══════════════════════════════════════════════════════════════════════════════

def make_rag_tool(vectorstore: Chroma) -> Tool:
    """Tool 1: RAG — search the indexed FastAPI documentation."""
    def rag_search(query: str) -> str:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        if not docs:
            return "No relevant documentation found for this query."
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"[Doc {i} - {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}")
        return "\n\n---\n\n".join(results)

    return Tool(
        name="search_fastapi_docs",
        description=(
            "Search the indexed FastAPI documentation and README. "
            "Use this for questions about FastAPI concepts, features, how-to guides, "
            "code examples from the official docs, installation, setup, and deployment. "
            "Do NOT use this for live GitHub data like issues, PRs, or recent commits."
        ),
        func=rag_search
    )


def make_issues_tool() -> Tool:
    """Tool 2: Fetch live GitHub issues."""
    def get_issues(query: str) -> str:
        try:
            headers = {"Accept": "application/vnd.github+json"}
            if GITHUB_TOKEN:
                headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
            params = {
                "q": f"{query} repo:{GITHUB_REPO} is:issue is:open",
                "sort": "updated",
                "per_page": 5
            }
            response = requests.get(
                "https://api.github.com/search/issues",
                headers=headers, params=params, timeout=10
            )
            data = response.json()
            if not data.get("items"):
                return f"No open issues found matching '{query}'"
            results = [f"Found {data['total_count']} matching issues. Top {len(data['items'])}:\n"]
            for issue in data["items"]:
                results.append(
                    f"#{issue['number']} — {issue['title']}\n"
                    f"  State: {issue['state']} | Comments: {issue['comments']}\n"
                    f"  Labels: {', '.join(l['name'] for l in issue['labels']) or 'none'}\n"
                    f"  URL: {issue['html_url']}\n"
                    f"  Body: {(issue['body'] or '')[:200]}..."
                )
            return "\n\n".join(results)
        except Exception as e:
            return f"Error fetching issues: {e}"

    return Tool(
        name="search_github_issues",
        description=(
            "Search live, real-time GitHub issues for the FastAPI repository. "
            "Use this for questions about current open bugs, feature requests, "
            "known issues with specific functionality, or community-reported problems. "
            "This fetches LIVE data from GitHub."
        ),
        func=get_issues
    )


def make_prs_tool() -> Tool:
    """Tool 3: Fetch recent GitHub pull requests."""
    def get_pull_requests(query: str) -> str:
        try:
            headers = {"Accept": "application/vnd.github+json"}
            if GITHUB_TOKEN:
                headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
            params = {
                "q": f"{query} repo:{GITHUB_REPO} is:pr is:open",
                "sort": "updated",
                "per_page": 5
            }
            response = requests.get(
                "https://api.github.com/search/issues",
                headers=headers, params=params, timeout=10
            )
            data = response.json()
            if not data.get("items"):
                prs  = github_get("pulls?state=open&sort=updated&per_page=5")
                data = {"items": prs, "total_count": len(prs)} if prs else {"items": [], "total_count": 0}
            if not data["items"]:
                return "No open PRs found."
            results = [f"Found {data['total_count']} PRs. Top {len(data['items'])}:\n"]
            for pr in data["items"]:
                results.append(
                    f"#{pr['number']} — {pr['title']}\n"
                    f"  URL: {pr['html_url']}\n"
                    f"  Body: {(pr.get('body') or '')[:200]}..."
                )
            return "\n\n".join(results)
        except Exception as e:
            return f"Error fetching PRs: {e}"

    return Tool(
        name="search_github_prs",
        description=(
            "Search live GitHub pull requests for the FastAPI repository. "
            "Use this for questions about recent changes, what features are being worked on, "
            "or active contributions. Fetches LIVE data from GitHub."
        ),
        func=get_pull_requests
    )


def make_file_tree_tool() -> Tool:
    """Tool 4: Browse the repo file structure."""
    def get_file_tree(path: str) -> str:
        try:
            path     = path.strip().strip("/") if path.strip() else ""
            endpoint = f"contents/{path}" if path else "contents"
            items    = github_get(endpoint)
            if isinstance(items, dict):
                return f"File: {items['path']} ({items.get('size', 0)} bytes)\nDownload: {items.get('download_url', 'N/A')}"
            lines = [f"📁 /{path or ''}\n"]
            for item in sorted(items, key=lambda x: (x["type"] != "dir", x["name"])):
                icon = "📁" if item["type"] == "dir" else "📄"
                lines.append(f"  {icon} {item['name']}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error browsing files: {e}"

    return Tool(
        name="browse_repo_files",
        description=(
            "Browse the live FastAPI GitHub repository file structure. "
            "Use this for questions about where specific files or modules are located, "
            "what's inside a directory, or the overall codebase structure. "
            "Pass a path like 'fastapi' or 'tests', or leave empty for the root."
        ),
        func=get_file_tree
    )


def make_repo_stats_tool() -> Tool:
    """Tool 5: Get repo metadata and stats."""
    def get_repo_stats(_: str) -> str:
        try:
            data = github_get("")
            return json.dumps({
                "name":        data.get("full_name"),
                "description": data.get("description"),
                "stars":       data.get("stargazers_count"),
                "forks":       data.get("forks_count"),
                "open_issues": data.get("open_issues_count"),
                "language":    data.get("language"),
                "license":     data.get("license", {}).get("name") if data.get("license") else None,
                "created":     data.get("created_at"),
                "last_push":   data.get("pushed_at"),
                "topics":      data.get("topics", [])
            }, indent=2)
        except Exception as e:
            return f"Error fetching repo stats: {e}"

    return Tool(
        name="get_repo_stats",
        description=(
            "Get live statistics and metadata about the FastAPI GitHub repository. "
            "Use this for questions about star count, forks, license, last update, "
            "or general repo information. Input is ignored — stats are always fetched fresh."
        ),
        func=get_repo_stats
    )


# ── Build Vector Store (reuse Demo 2 Chroma if it exists) ────────────────────
def get_or_build_vectorstore() -> Chroma:
    print("⏳ Initialising local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("   ✅ Embedding model ready\n")

    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print("📦 Reusing Chroma vector store from Demo 2...")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        print(f"   ✅ {vectorstore._collection.count()} vectors loaded\n")
        return vectorstore

    print("📥 Building vector store from scratch...")
    urls = {
        "README":           "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
        "docs/features":    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/features.md",
        "docs/first_steps": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/first-steps.md",
        "docs/path_params": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-params.md",
        "docs/deployment":  "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/deployment/index.md",
    }
    documents = []
    for name, url in urls.items():
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            documents.append(Document(page_content=r.text, metadata={"source": name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"   ✅ Built with {vectorstore._collection.count()} vectors\n")
    return vectorstore


# ── Build the Agent ───────────────────────────────────────────────────────────
def build_agent(vectorstore: Chroma) -> AgentExecutor:
    """
    Key architectural shift from Demo 2:

    Demo 2: Chain → always runs RAG → always generates answer
    Demo 3: Agent → reads tool descriptions → decides which tool(s) to call

    Because Groq's Llama supports OpenAI-compatible function calling, we can
    use create_openai_functions_agent — the agent selects tools via structured
    JSON rather than ReAct text loops, giving more reliable tool selection.

    This is exactly what MCP enables at scale:
      Tools advertise their capabilities → Agents auto-discover and use them.
    """
    llm = ChatGroq(
        model=CHAT_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=1024,
    )

    tools = [
        make_rag_tool(vectorstore),
        make_issues_tool(),
        make_prs_tool(),
        make_file_tree_tool(),
        make_repo_stats_tool(),
    ]

    print("🔧 Tools registered with agent:")
    for tool in tools:
        print(f"   • {tool.name}")
    print()

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True
    )

    # Standard OpenAI functions agent prompt from LangChain hub
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )

    return agent


# ── Main Chat Loop ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🤖 GitHub Repo Assistant — Demo 3: RAG + LangChain + MCP")
    print("  📌 Embeddings: local (sentence-transformers)")
    print("  📌 Agent:      Groq API  (llama-3.1-8b-instant)")
    print("=" * 60)
    print()

    vectorstore = get_or_build_vectorstore()
    agent       = build_agent(vectorstore)

    print("✅ Agent ready! I now have BOTH static docs AND live GitHub access.\n")
    print("   Try these (notice how the agent picks the right tool):\n")
    print("   📚 Doc questions  (→ uses RAG tool):")
    print("      - 'How do I define a path parameter?'")
    print("      - 'Explain FastAPI dependency injection'")
    print()
    print("   🐛 Live issues    (→ uses GitHub issues tool):")
    print("      - 'Are there any open issues about WebSockets?'")
    print()
    print("   🔀 Live PRs       (→ uses GitHub PR tool):")
    print("      - 'What features are being worked on right now?'")
    print()
    print("   📊 Stats          (→ uses repo stats tool):")
    print("      - 'How many stars does FastAPI have?'")
    print()
    print("   🧠 Multi-tool + memory:")
    print("      - 'Compare what the docs say about routing vs open routing issues'")
    print()
    print("   [verbose=True so you can SEE the agent reasoning and tool selection]")
    print("   [Type 'quit' to exit]\n")
    print("-" * 60)

    while True:
        query = input("\n🧑 You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        print()
        result = agent.invoke({"input": query})
        print(f"\n🤖 Final Answer: {result['output']}")


if __name__ == "__main__":
    main()
