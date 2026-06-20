import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from src.embed import get_embeddings_model
from src.store import load_index
from src.retrieve import retrieve_chunks
from src.generate import generate_answer

load_dotenv()

app = Flask(__name__)

print("Loading embeddings model...")
embeddings = get_embeddings_model()
print("Loading FAISS index...")
vector_store = load_index(embeddings, "index")
print("Ready.")

RELEVANCE_THRESHOLD = 1.5

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]
    k = data.get("k", 4)
    strategy = data.get("strategy", "similarity")

    retrieved = retrieve_chunks(vector_store, question, k=k, strategy=strategy)
    answer = generate_answer(question, retrieved)

    best_score = retrieved[0][1] if retrieved else float("inf")
    if best_score > RELEVANCE_THRESHOLD:
        return jsonify({
            "answer": "I don't have information about that in my support docs. Please contact support@acmestore.example for help with this question.",
            "sources": [],
            "out_of_scope": True,
        })

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "preview": doc.page_content[:200],
        }
        for doc, score in retrieved
    ]

    return jsonify({"answer": answer, "sources": sources})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)