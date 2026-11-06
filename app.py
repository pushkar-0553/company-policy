import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from rag_engine import answer_policy_question, init_vector_store

load_dotenv()

app = Flask(__name__)

# Warm up / optionally initialize vector store on startup
@app.before_request
def startup_check():
    # Only run once if needed
    if not getattr(app, "_has_warmed_up", False):
        try:
            init_vector_store()
        except Exception as e:
            app.logger.warning(f"Vector store warmup warning (can be configured later): {e}")
        app._has_warmed_up = True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    result = answer_policy_question(user_message)
    return jsonify({
        "reply": result.get("answer", ""),
        "sources": result.get("sources", [])
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Company Policy Assistant"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

