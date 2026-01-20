import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


def retrieve_top_k(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    documents: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:

    # Compute cosine similarity
    scores = cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]

    # Get indices of top-k scores
    top_k_indices = scores.argsort()[-top_k:][::-1]

    # Return documents with scores
    results = [(documents[i], float(scores[i])) for i in top_k_indices]

    return results
