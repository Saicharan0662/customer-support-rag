import os
from groq import Groq

SYSTEM_PROMPT = """You are a customer support assistant. Answer the user's question using only the context provided below. If the context does not contain enough information to answer, say so clearly. Cite the source filename once at the end of your answer, not after every sentence."""

def generate_answer(query: str, retrieved):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context_blocks = []
    for i, (doc, score) in enumerate(retrieved, start=1):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[Source {i}: {source}]\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content