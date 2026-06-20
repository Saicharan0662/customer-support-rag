from dotenv import load_dotenv
from src.ingest import load_documents
from src.chunk import chunk_documents
from src.embed import get_embeddings_model
from src.store import build_and_save_index

load_dotenv()

DOCS_DIR = "data/docs"
INDEX_DIR = "index"

def main():
    documents = load_documents(DOCS_DIR)
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
    embeddings = get_embeddings_model()
    build_and_save_index(chunks, embeddings, INDEX_DIR)
    print("Index built successfully.")

if __name__ == "__main__":
    main()