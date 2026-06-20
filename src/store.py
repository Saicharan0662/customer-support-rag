from pathlib import Path
from langchain_community.vectorstores import FAISS

def build_and_save_index(chunks, embeddings, index_dir: str):
    vector_store = FAISS.from_documents(chunks, embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"Saved FAISS index to {index_dir}")
    return vector_store

def load_index(embeddings, index_dir: str):
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )