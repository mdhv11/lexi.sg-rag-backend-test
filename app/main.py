from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline

app = FastAPI(title="Lexi RAG Backend")

rag = RAGPipeline()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    answer, citations = rag.answer_query(request.query)
    return {
        "answer": answer,
        "citations": citations
    }
