from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer
    Keeps embedding logic isolated so we can
    swap to Gemini later without touching FAISS code
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Convert list of strings to embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype="float32")

    def embed_query(self, query):
        """
        Embed a single query
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return np.array([embedding], dtype="float32")
