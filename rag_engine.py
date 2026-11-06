import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Cache instances
_client = None
_vector_store = None
_retriever = None
_llm = None

def get_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def get_qdrant_client():
    global _client
    if _client is None:
        qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
        qdrant_key = os.getenv("QDRANT_KEY")
        if not qdrant_url or not qdrant_key:
            return None
        _client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    return _client

def get_embeddings():
    api_key = get_api_key()
    if not api_key:
        return None
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

def get_llm():
    global _llm
    if _llm is None:
        api_key = get_api_key()
        if not api_key:
            return None
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        _llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2
        )
    return _llm

def init_vector_store():
    global _vector_store, _retriever
    if _vector_store is not None and _retriever is not None:
        return _retriever

    client = get_qdrant_client()
    if not client:
        return None

    embeddings = get_embeddings()
    if not embeddings:
        return None

    collection_name = os.getenv("QDRANT_COLLECTION", "company_policy")

    # Check if collection exists and has points
    needs_indexing = False
    try:
        col_info = client.get_collection(collection_name)
        if col_info.vectors_count == 0 and col_info.points_count == 0:
            needs_indexing = True
    except Exception:
        # Collection does not exist yet
        needs_indexing = True

    if needs_indexing:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        policy_path = os.path.join(base_dir, "policy.txt")
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"policy.txt not found at {policy_path}")

        with open(policy_path, "r", encoding="utf-8") as f:
            text_data = f.read()

        documents = [Document(page_content=text_data)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Re-create/create collection and upload chunks
        _vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
            api_key=os.getenv("QDRANT_KEY"),
            collection_name=collection_name,
            force_recreate=True
        )
    else:
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )

    _retriever = _vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    return _retriever

def answer_policy_question(question: str) -> dict:
    api_key = get_api_key()
    if not api_key:
        return {
            "answer": "⚠️ Gemini API key not configured. Please set GEMINI_API_KEY in your .env or Render environment variables.",
            "sources": []
        }

    qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    qdrant_key = os.getenv("QDRANT_KEY")
    if not qdrant_url or not qdrant_key:
        return {
            "answer": "⚠️ Qdrant credentials not configured. Please set QDRANT_CLUSTER_ENDPOINT and QDRANT_KEY.",
            "sources": []
        }

    try:
        retriever = init_vector_store()
        if not retriever:
            return {
                "answer": "⚠️ Unable to initialize vector database. Check your credentials.",
                "sources": []
            }

        documents = retriever.invoke(question)

        if not documents:
            return {
                "answer": "No relevant company policy information was found regarding your question.",
                "sources": []
            }

        context = "\n\n".join(doc.page_content for doc in documents)

        prompt = f"""You are an AI Company Policy Assistant.

Answer the employee's question using ONLY the company policy information provided below.
If the answer is not available in the policy, clearly say that the information is not available in the company policy.
Keep your response clear, professional, and easy to read.

Company Policy Context:
{context}

Employee Question:
{question}

Answer:"""

        llm = get_llm()
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        sources = [doc.page_content for doc in documents]
        return {
            "answer": answer_text,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "sources": []
        }

