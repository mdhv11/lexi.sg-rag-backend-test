import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbedStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.metadata = []

    def embed_and_store(self, docs):
        all_chunks = []
        for filename, text in docs:
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            embeddings = self.model.encode(chunks)
            self.index.add(np.array(embeddings).astype("float32"))
            self.metadata.extend(
                [{"text": chunk, "source": filename} for chunk in chunks])

    def search(self, query, top_k=3):
        query_vector = self.model.encode([query]).astype("float32")
        D, I = self.index.search(query_vector, top_k)
        return [self.metadata[i] for i in I[0]]
