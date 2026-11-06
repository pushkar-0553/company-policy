import os
import sys
from dotenv import load_dotenv

load_dotenv()

def run_ingestion():
    print("=" * 55)
    print("  Company Policy Vector Store Ingestion (FastEmbed)  ")
    print("=" * 55)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("\n⚠️ Note: GROQ_API_KEY is not set in .env.")
        print("FastEmbed can still index documents without an API key!")
        print("However, to test the final answer generation, get a free key at: https://console.groq.com/keys")

    print("\n1. Initializing FastEmbed & Qdrant...")
    from rag_engine import init_vector_store, answer_policy_question
    
    try:
        retriever = init_vector_store()
        if not retriever:
            print("❌ Failed to initialize vector store. Check your Qdrant credentials.")
            sys.exit(1)
        print("✅ Vector store initialized and indexed into Qdrant successfully!")
    except Exception as e:
        print(f"❌ Error during vector store initialization: {e}")
        sys.exit(1)

    if groq_key:
        print("\n2. Testing sample retrieval & Groq answer...")
        test_question = "What is the annual paid leave allotment?"
        print(f"Query: '{test_question}'")
        result = answer_policy_question(test_question)
        print("\nGroq Assistant Response:")
        print("-" * 55)
        print(result.get("answer", "No response."))
        print("-" * 55)
        print("\n🎉 Everything is working properly!")
    else:
        print("\n✅ Embeddings and Qdrant indexing completed! Add GROQ_API_KEY to test chat responses.")

if __name__ == "__main__":
    run_ingestion()
