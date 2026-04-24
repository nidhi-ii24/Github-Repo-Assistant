"""
╔══════════════════════════════════════════════════════════════╗
║         DEMO 2 — RAG + LangChain                            ║
║         GitHub Repo Assistant — FastAPI Docs                 ║
║                                                              ║
║  Stack:  LangChain + Chroma + sentence-transformers + Groq  ║
║  Flow:   Same RAG — but now orchestrated + stateful          ║
╚══════════════════════════════════════════════════════════════╝

What's new vs Demo 1:
  ✅ Chroma replaces FAISS         → persistent, survives restarts
  ✅ LangChain chains              → less boilerplate, composable
  ✅ ConversationMemory            → multi-turn conversations work
  ✅ ConversationalRetrievalChain  → prompt augmentation automated

What breaks at the end (motivates Demo 3):
  - Tools are hardcoded — can't fetch live GitHub data
  - Every new integration needs custom Python glue code
  - No standardized way to plug in external capabilities

NOTE on model choices:
  Embeddings → sentence-transformers (local, free, no API calls)
  Chat       → Groq API (llama-3.1-8b-instant, free tier, no credit card)
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

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "fastapi_docs"

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # local, free
CHAT_MODEL    = "llama-3.1-8b-instant"                    # Groq free tier

TOP_K         = 4
MEMORY_WINDOW = 5   # remember last 5 exchanges


# ── Step 1: Fetch + Load Documents ───────────────────────────────────────────
def load_documents() -> list[Document]:
    """
    LangChain works with Document objects — text + metadata.
    Same sources as Demo 1, but now structured as LangChain Documents.

    In production you'd use LangChain's built-in loaders:
      - GitLoader      → clone and load entire repos
      - GitHubLoader   → load issues, PRs, files via API
      - WebBaseLoader  → load any webpage
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
    """
    RecursiveCharacterTextSplitter is smarter than our manual chunker in Demo 1.

    It tries to split on: paragraphs → sentences → words → characters
    This means chunks are more likely to end at natural boundaries.
    """
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
    """
    Uses HuggingFaceEmbeddings (local sentence-transformers) — zero API cost.

    Chroma persists to disk so you only embed once.
    Delete ./chroma_db/ to force a rebuild.

    Chroma vs FAISS (from Demo 1):
      FAISS  → in-memory only, no metadata filtering, raw numpy
      Chroma → persists to disk, filter by metadata, managed Documents
    """
    print("⏳ Initialising local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},          # change to "cuda" if you have a GPU
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
    """
    Uses ChatGroq (llama-3.1-8b-instant) — free tier, no credit card.

    This is where LangChain's power becomes obvious vs Demo 1.
    ConversationalRetrievalChain does everything automatically:
      - Rephrases follow-up questions using chat history
        (e.g. "tell me more" → "tell me more about path parameters")
      - Injects conversation memory
      - Handles the retrieval → augmentation → generation pipeline
    """
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
Answer the question using ONLY the context provided.
If the answer isn't in the context, say "I don't have enough information in the indexed docs."
Be concise and helpful. If showing code, format it properly.

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

    return chain


# ── Main Chat Loop ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🤖 GitHub Repo Assistant — Demo 2: RAG + LangChain")
    print("  📌 Embeddings: local (sentence-transformers)")
    print("  📌 Chat:       Groq API  (llama-3.1-8b-instant)")
    print("=" * 60)
    print()

    documents   = load_documents()
    chunks      = split_documents(documents)
    vectorstore = get_vector_store(chunks)
    chain       = build_chain(vectorstore)

    print("✅ Ready! This time I remember our conversation.\n")
    print("   Try this multi-turn sequence:")
    print("   1. 'What are path parameters in FastAPI?'")
    print("   2. 'Can you show me an example?'          ← follow-up, needs memory")
    print("   3. 'What did I first ask you?'            ← tests memory")
    print("   4. 'What are the open GitHub issues?'     ← this will FAIL (no live data)\n")
    print("   [Type 'quit' to exit]\n")
    print("-" * 60)

    while True:
        query = input("\n🧑 You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result  = chain.invoke({"question": query})
        sources = result.get("source_documents", [])
        if sources:
            print(f"\n📚 Sources used: {', '.join(set(s.metadata['source'] for s in sources))}")

        print(f"\n🤖 Assistant: {result['answer']}")

        # ── THE CLIFFHANGER ──
        # Try: "What are the open issues about WebSockets?" → will fail → needs Demo 3


if __name__ == "__main__":
    main()
