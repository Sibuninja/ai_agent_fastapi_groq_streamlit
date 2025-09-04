# rag_engine.py
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from utils_rag import extract_text_from_pdf, chunk_text

# load embedding model once (might download on first run)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class RAGPipeline:
    def __init__(self):
        # keep TF-IDF out until you actually use it
        self.text_chunks = []
        self.nn_model = None
        self.embeddings = None
        self.embedding_dim = None

    def prepare_doc(self, file_path):
        """
        Extract text, chunk it, compute embeddings and fit nearest-neighbors.
        """
        print("🔧 Preparing document for RAG...")

        raw_text = extract_text_from_pdf(file_path)
        if not raw_text or not raw_text.strip():
            raise ValueError("No text extracted from PDF.")

        self.text_chunks = chunk_text(raw_text)
        if not self.text_chunks:
            raise ValueError("No text chunks produced from PDF.")

        print("📄 Page Text (preview):", (self.text_chunks[0][:300] + "...") if self.text_chunks else "❌ No text extracted")

        # Compute embeddings (returns numpy array shape (n_chunks, dim))
        self.embeddings = embedding_model.encode(self.text_chunks, show_progress_bar=False)
        if self.embeddings is None or len(self.embeddings) == 0:
            raise RuntimeError("Failed to compute embeddings.")

        # Optionally normalize embeddings for cosine similarity using sklearn
        # (NearestNeighbors with metric='cosine' works without normalization but normalization is stable)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        self.embedding_dim = self.embeddings.shape[1]

        # Fit NearestNeighbors. We'll call kneighbors with a dynamic n_neighbors later.
        self.nn_model = NearestNeighbors(metric="cosine")
        self.nn_model.fit(self.embeddings)

    def retrieve_similar_chunks(self, query, top_k=3):
        """
        Return top_k most similar text chunks for the query.
        """
        if self.nn_model is None or self.embeddings is None or not self.text_chunks:
            raise RuntimeError("RAG pipeline not prepared. Call prepare_doc(file_path) first.")

        query_embedding = embedding_model.encode([query], show_progress_bar=False)
        # normalize
        qnorm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        qnorm[qnorm == 0] = 1.0
        query_embedding = query_embedding / qnorm

        k = min(top_k, len(self.text_chunks))
        distances, indices = self.nn_model.kneighbors(query_embedding, n_neighbors=k)
        return [self.text_chunks[i] for i in indices[0]]

    def query_doc(self, question):
        """
        Build a prompt using retrieved contexts for the question.
        """
        chunks = self.retrieve_similar_chunks(question, top_k=3)
        context = "\n\n".join(chunks)
        prompt = f"Answer the following question using the context. If the answer is not in the context, say 'I don't know':\n\nContext:\n{context}\n\nQuestion: {question}"
        return prompt
