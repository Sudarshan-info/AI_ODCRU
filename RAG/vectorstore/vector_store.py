import numpy as np
from typing import List


class VectorStore:
    def __init__(self):
        self.embeddings = None  # numpy array
        self.documents = []  # list of text chunks

    def add(self, embeddings: np.ndarray, documents: List[str]):
        self.embeddings = embeddings
        self.documents = documents

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings

    def get_documents(self) -> List[str]:
        return self.documents
