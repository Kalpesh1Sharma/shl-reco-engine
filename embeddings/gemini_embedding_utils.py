"""
DEPRECATED / EXPERIMENTAL

Gemini embeddings were explored but NOT used in the final system
due to free-tier quota and bulk embedding limitations.

Kept only for experimentation / future work reference.
"""


import google.generativeai as genai
import numpy as np
import os

# Expect API key in environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/embedding-001"


class GeminiEmbeddingModel:
    def embed_texts(self, texts):
        embeddings = []
        for t in texts:
            result = genai.embed_content(
                model=MODEL_NAME,
                content=t,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])

        return np.array(embeddings, dtype="float32")

    def embed_query(self, query):
        result = genai.embed_content(
            model=MODEL_NAME,
            content=query,
            task_type="retrieval_query"
        )
        return np.array([result["embedding"]], dtype="float32")
