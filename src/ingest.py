from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_documents(docs_dir: str):
    docs_path = Path(docs_dir)
    documents = []

    for file_path in docs_path.rglob("*"):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} documents from {docs_dir}")
    return documents