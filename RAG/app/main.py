import sys
import os

# Add the project root (parent of app/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import pickle


from loaders.document_loader import load_documents
from splitters.text_splitter import split_documents
from embeddings.embedding_model import get_embedding_model
from retriever.similarity_retriever import retrieve_top_k
from memory.memory import get_memory
from config.settings import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TOP_K,
    USE_MEMORY,
)

CHUNKS_FILE = PROCESSED_DATA_PATH


def main():
    documents = []
    print("\n=== Semantic Retrieval System (Cosine Similarity) ===")

    # ------------------------
    # Load or create chunks
    # ------------------------
    if os.path.exists(CHUNKS_FILE):
        print(f"Loading processed chunks from {CHUNKS_FILE}...")
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks.")
    else:
        print("Loading raw documents...")
        documents = load_documents(RAW_DATA_PATH)
        print(f"Loaded {len(documents)} documents.")

        print("Splitting documents into chunks...")
        chunks = split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        # Save chunks for future use
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        print(f"Saved chunks to {CHUNKS_FILE}.")

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Embedding model loaded.")
    document_embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    memory = get_memory() if USE_MEMORY else None
    if memory:
        print("Conversation memory enabled.\n")

    print("System ready. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Exiting system. Goodbye!")
            break

        query_embedding = embedding_model.encode(query)

        results = retrieve_top_k(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            documents=chunks,
            top_k=TOP_K,
        )

        # Save conversation context if memory is enabled
        if memory:
            memory.save_context(
                {"input": query}, {"output": [text for text, _ in results]}
            )

        # Display retrieved chunks
        print("\n--- Retrieved Chunks ---")
        for i, (text, score) in enumerate(results, 1):
            print(f"\nChunk {i} (score={score:.3f}):\n{text}")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
