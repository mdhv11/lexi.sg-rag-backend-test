from app.document_loader import load_documents
from app.embed_store import EmbedStore
from app.model import LocalLLM


class RAGPipeline:
    def __init__(self):
        self.docs = load_documents()
        self.store = EmbedStore()
        self.store.embed_and_store(self.docs)
        self.llm = LocalLLM()

    def answer_query(self, query):
        top_chunks = self.store.search(query)
        context = "\n".join([chunk["text"] for chunk in top_chunks])
        prompt = f"""You are a legal assistant. Given the query: "{query}", and the legal context below, answer concisely and cite relevant text.

Context:
{context}

Answer:"""
        answer = self.llm.generate(prompt)
        citations = [{"text": chunk["text"], "source": chunk["source"]}
                     for chunk in top_chunks]
        return answer, citations
