def retrieve_chunks(vector_store, query: str, k: int = 4, strategy: str = "similarity"):
    if strategy == "mmr":
        results = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
        return [(doc, 0.0) for doc in results]
    results = vector_store.similarity_search_with_score(query, k=k)
    return results