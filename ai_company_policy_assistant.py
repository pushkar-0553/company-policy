import os
from rag_engine import answer_policy_question

if __name__ == "__main__":
    print("==================================================")
    print("       AI Company Policy Assistant (Terminal)      ")
    print("==================================================")
    print("Type your question below or type 'exit' to quit.\n")

    while True:
        try:
            question = input("\nAsk a policy question (or type 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Company Policy Assistant...")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Exiting Company Policy Assistant...")
            break

        print("\nThinking...")
        result = answer_policy_question(question)
        print("\nAnswer:")
        print(result.get("answer", "No answer generated."))
