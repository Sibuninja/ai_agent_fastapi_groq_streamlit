import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from utils_rag import extract_text_from_pdf, chunk_text

# Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class RAGPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.text_chunks = []
        self.nn_model = None
        self.embeddings = None

    def prepare_doc(self, file_path):
        print("🔧 Preparing document for RAG...")

        raw_text = extract_text_from_pdf(file_path)
        self.text_chunks = chunk_text(raw_text)
        print("📄 Page Text:", self.text_chunks[0][:300] + "..." if self.text_chunks else "❌ No text extracted")

        # Generate embeddings using sentence-transformers
        self.embeddings = embedding_model.encode(self.text_chunks)

        # Fit NearestNeighbors on embeddings
        self.nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.nn_model.fit(self.embeddings)

    def retrieve_similar_chunks(self, query, top_k=3):
        query_embedding = embedding_model.encode([query])
        k = min(top_k, len(self.text_chunks))  # Prevent exceeding available samples
        distances, indices = self.nn_model.kneighbors(query_embedding, n_neighbors=k)
        return [self.text_chunks[i] for i in indices[0]]

    def query_doc(self, question):
        context = "\n".join(self.retrieve_similar_chunks(question))
        prompt = f"Answer the following question using the context:\n\nContext:\n{context}\n\nQuestion: {question}"
        return prompt
