from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME


def get_embedding_model():
    """
    Load the sentence transformer embedding model.

    Returns:
        model (SentenceTransformer): Embedding model
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def embed_chunks(model, chunks):
    """
    Generate embeddings for a list of text chunks.

    Args:
        model (SentenceTransformer): Loaded embedding model
        chunks (List[str]): List of text chunks

    Returns:
        List[List[float]]: List of vector embeddings
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings
