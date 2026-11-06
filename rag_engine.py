import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Load environment variables
load_dotenv()

# FastEmbed wrapper conforming to LangChain Embeddings interface
class FastEmbedModel(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vec)) for vec in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(self.model.embed([text]))))

# Cache instances
_client = None
_vector_store = None
_retriever = None
_llm = None
_embeddings = None

def get_groq_api_key():
    return os.getenv("GROQ_API_KEY")

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
    global _embeddings
    if _embeddings is None:
        # FastEmbed runs locally using ONNX, zero API key needed
        _embeddings = FastEmbedModel(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings

def get_llm():
    global _llm
    if _llm is None:
        groq_key = get_groq_api_key()
        if groq_key:
            from langchain_groq import ChatGroq
            model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
            _llm = ChatGroq(
                model=model_name,
                groq_api_key=groq_key,
                temperature=0.2
            )
        else:
            # Fallback to Gemini if GEMINI_API_KEY is provided
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                from langchain_google_genai import ChatGoogleGenerativeAI
                _llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=gemini_key,
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

        # Create/re-create collection in Qdrant with FastEmbed vectors (384-dim)
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
    groq_key = get_groq_api_key()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not groq_key and not gemini_key:
        return {
            "answer": "⚠️ Groq API key is not configured. Please set GROQ_API_KEY in your .env or Render environment variables (Get free key at: https://console.groq.com/keys).",
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
                "answer": "⚠️ Unable to initialize Qdrant vector database. Please verify your credentials.",
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
If the answer is not available in the policy, clearly state that the information is not available in the company policy.
Keep your answer clear, polite, and well-structured.

Company Policy Context:
{context}

Employee Question:
{question}

Answer:"""

        llm = get_llm()
        if not llm:
            return {
                "answer": "⚠️ Could not initialize the LLM. Please verify your GROQ_API_KEY.",
                "sources": []
            }

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
