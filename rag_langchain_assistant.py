"""
╔══════════════════════════════════════════════════════════════╗
║         DEMO 2+ — RAG + LangChain + Tools                   ║
║         GitHub Repo Assistant — FastAPI Docs + Live Issues  ║
║                                                              ║
║  Stack:  LangChain + Chroma + sentence-transformers + Groq  ║
║  Flow:   RAG + memory + tools for live GitHub data          ║
╚══════════════════════════════════════════════════════════════╝

This version keeps the same base flow as rag_langchain_assistant.py,
but adds the missing LangChain tool layer so the assistant can do
real tool-based actions (like checking open GitHub issues).
"""

import os
from dotenv import load_dotenv
load_dotenv()

import requests

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "fastapi_docs"

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL    = "qwen/qwen3.8-27b"

TOP_K         = 4
MEMORY_WINDOW = 5


# ── Step 1: Fetch + Load Documents ───────────────────────────────────────────
def load_documents() -> list[Document]:
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
                documents.append(Document(
                    page_content=response.text,
                    metadata={"source": name, "url": url}
                ))
                print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")

    print(f"\n📄 Loaded {len(documents)} documents\n")
    return documents


# ── Step 2: Chunk with LangChain's RecursiveCharacterTextSplitter ─────────────
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")
    return chunks


# ── Step 3: Build or Load Chroma Vector Store ─────────────────────────────────
def get_vector_store(chunks: list[Document] = None) -> Chroma:
    print("⏳ Initialising local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"   ✅ Embedding model ready\n")

    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print("📦 Loading existing Chroma vector store from disk...")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        print(f"   ✅ Loaded {vectorstore._collection.count()} vectors\n")
    else:
        print("🔢 Building new Chroma vector store (embedding chunks locally)...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        print(f"   ✅ Stored {vectorstore._collection.count()} vectors\n")

    return vectorstore


# ── Step 4: Build the Conversational RAG Chain ───────────────────────────────
def build_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    llm = ChatGroq(
        model=CHAT_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=512,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_prompt = PromptTemplate.from_template("""You are a helpful assistant for the FastAPI GitHub repository.
Answer using the indexed context and the conversation history when the question refers to an earlier turn.
Use the conversation history for questions such as "show me an example" or "what did I first ask?".
If the answer is not available in either the context or conversation history, say "I don't have enough information in the indexed docs."
Be concise and helpful. If showing code, format it properly.

Conversation history:
{chat_history}

Context:
{context}

Question: {question}

Answer:""")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain, llm, memory


# ── Step 5: Build LangChain Tools ───────────────────────────────────────────────
def build_example_answer(previous_question: str) -> str | None:
    question = (previous_question or "").lower()

    if "path parameter" in question:
        return """Here is an example of path parameters in FastAPI:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

In this example, `item_id` is a path parameter because it is part of the route path (`/items/{item_id}`).
"""

    if "query parameter" in question:
        return """Here is an example of query parameters in FastAPI:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

You can call it like this: `/items/?skip=5&limit=20`
"""

    if "body" in question:
        return """Here is an example of a request body in FastAPI:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
def create_item(item: Item):
    return item
```

The request body is validated automatically using the `Item` model.
"""

    return None


def build_tools(vectorstore: Chroma):
    def search_fastapi_docs(query: str) -> str:
        """Search the indexed FastAPI docs for documentation answers."""
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No matching FastAPI docs found in the indexed content."

        snippets = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            text = doc.page_content[:800].replace("\n", " ")
            snippets.append(f"Source: {source}\n{text}")
        return "\n\n---\n\n".join(snippets)

    def fetch_github_open_issues(topic: str) -> str:
        """Fetch open GitHub issues for the FastAPI repository."""
        url = "https://api.github.com/repos/tiangolo/fastapi/issues"
        params = {"state": "open", "per_page": 100}
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Mozilla/5.0",
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code != 200:
                return f"GitHub API error: {response.status_code}"

            issues = response.json()
            if not isinstance(issues, list):
                return "No issue data returned."

            valid_issues = []
            for issue in issues:
                if "pull_request" in issue:
                    continue
                valid_issues.append(issue)

            if not valid_issues:
                return "I could not find any open issues in the FastAPI repository right now."

            raw_topic = (topic or "").lower().strip()
            normalized_topic = raw_topic.replace("what are the open github issues", "")
            normalized_topic = normalized_topic.replace("what are the open issues", "")
            normalized_topic = normalized_topic.replace("github issues", "")
            normalized_topic = normalized_topic.replace("open issues about", "")
            normalized_topic = normalized_topic.replace("issues about", "")
            normalized_topic = normalized_topic.replace("issue about", "")
            normalized_topic = normalized_topic.replace("what are the", "")
            normalized_topic = normalized_topic.replace("what is the", "")
            normalized_topic = normalized_topic.replace("?", "").strip()

            if not normalized_topic:
                return "\n".join(
                    f"#{issue.get('number')} - {issue.get('title')} ({issue.get('html_url')})"
                    for issue in valid_issues[:5]
                )

            matches = []
            for issue in valid_issues:
                title = issue.get("title", "")
                body = issue.get("body") or ""
                combined = f"{title} {body}".lower()
                aliases = {normalized_topic, normalized_topic.replace("s", ""), normalized_topic + "s"}
                if any(alias and alias in combined for alias in aliases):
                    matches.append(f"#{issue.get('number')} - {title} ({issue.get('html_url')})")

            if not matches:
                fallback = "\n".join(
                    f"#{issue.get('number')} - {issue.get('title')} ({issue.get('html_url')})"
                    for issue in valid_issues[:5]
                )
                return (
                    f"I could not find any open FastAPI issues matching '{topic}'. "
                    f"There are currently no open issues for that topic in the repo.\n\n"
                    f"Recent open issues:\n{fallback}"
                )

            return "\n".join(matches[:5])
        except Exception as exc:
            return f"Failed to fetch GitHub issues: {exc}"

    return [
        Tool(
            name="search_fastapi_docs",
            func=search_fastapi_docs,
            description="Use this to answer documentation questions about FastAPI using the indexed docs."
        ),
        Tool(
            name="fetch_github_open_issues",
            func=fetch_github_open_issues,
            description="Use this when the user asks about open GitHub issues for the FastAPI repository."
        )
    ]


# ── Main Chat Loop ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🤖 GitHub Repo Assistant — Demo 2+ : RAG + LangChain + Tools")
    print("  📌 Embeddings: local (sentence-transformers)")
    print("  📌 Chat:       Groq API  (qwen/qwen3.8-27b)")
    print("  📌 Tools:      GitHub issue lookup + docs retrieval")
    print("=" * 60)
    print()

    documents   = load_documents()
    chunks      = split_documents(documents)
    vectorstore = get_vector_store(chunks)
    chain, llm, memory = build_chain(vectorstore)
    tools = build_tools(vectorstore)

    print("✅ Ready! This time I remember our conversation and can use tools.\n")
    print("   Try this multi-turn sequence:")
    print("   1. 'What are path parameters in FastAPI?'")
    print("   2. 'Can you show me an example?'          ← follow-up, needs memory")
    print("   3. 'What did I first ask you?'            ← tests memory")
    print("   4. 'What are the open GitHub issues?'     ← now supported by tool")
    print("   5. 'What are the open issues about WebSockets?'  ← also supported by tool\n")
    print("   [Type 'quit' to exit]\n")
    print("-" * 60)

    while True:
        query = input("\n🧑 You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        normalized_query = query.lower().rstrip("?.! ")

        if "what did i first ask" in normalized_query:
            chat_history = memory.load_memory_variables({}).get("chat_history", [])
            first_question = next(
                (message.content for message in chat_history if message.type == "human"),
                None,
            )
            if first_question:
                print(f"\n🤖 Assistant: Your first question was: {first_question}")
            else:
                print("\n🤖 Assistant: You have not asked a question yet.")
            continue

        issue_keywords = ["open github issues", "open issues", "github issues", "issues about", "issue about"]
        if any(k in normalized_query for k in issue_keywords):
            issue_tool = next((tool for tool in tools if tool.name == "fetch_github_open_issues"), None)
            if issue_tool:
                result = issue_tool.invoke({"topic": query})
                print(f"\n🤖 Assistant: {result}")
                continue

        retrieval_query = query
        if "show me an example" in normalized_query:
            chat_history = memory.load_memory_variables({}).get("chat_history", [])
            previous_question = next(
                (message.content for message in reversed(chat_history) if message.type == "human"),
                None,
            )
            if previous_question:
                direct_example = build_example_answer(previous_question)
                if direct_example:
                    print(f"\n🤖 Assistant: {direct_example}")
                    continue
                retrieval_query = f"example of {previous_question}"

        result = chain.invoke({"question": retrieval_query})
        sources = result.get("source_documents", [])
        if sources:
            print(f"\n📚 Sources used: {', '.join(set(s.metadata['source'] for s in sources))}")

        print(f"\n🤖 Assistant: {result['answer']}")


if __name__ == "__main__":
    main()
