import os
import sys
from dotenv import load_dotenv

load_dotenv()

def run_ingestion():
    print("=" * 50)
    print("      Company Policy Vector Store Ingestion       ")
    print("=" * 50)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY is not set.")
        print("Please open .env and set GEMINI_API_KEY=your_key")
        print("Get your free key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    print("\n1. Connecting to Qdrant...")
    from rag_engine import init_vector_store, answer_policy_question
    
    try:
        retriever = init_vector_store()
        if not retriever:
            print("❌ Failed to initialize vector store. Check your Qdrant credentials.")
            sys.exit(1)
        print("✅ Vector store initialized and indexed successfully!")
    except Exception as e:
        print(f"❌ Error during vector store initialization: {e}")
        sys.exit(1)

    print("\n2. Testing sample retrieval...")
    test_question = "What is the annual paid leave allotment?"
    print(f"Query: '{test_question}'")
    result = answer_policy_question(test_question)
    print("\nAssistant Response:")
    print("-" * 50)
    print(result.get("answer", "No response."))
    print("-" * 50)
    print("\n🎉 Everything is working properly!")

if __name__ == "__main__":
    run_ingestion()
