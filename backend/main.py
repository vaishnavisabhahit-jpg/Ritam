import os
import json
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Ensure parent path resolution for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.safety import (
    check_emergency_intent,
    verify_grounded_quotes,
    verify_numerical_consistency,
    calculate_readability
)
from backend.agents.llm import generate_medical_answer

load_dotenv()

app = FastAPI(
    title="Ritam Medical AI API",
    description="Verified FDA Medical Intelligence Engine Backend",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# Initialize Chroma Vector Database Connection
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_collection("drug_labels")
except Exception as e:
    print(f"Warning: Vector database connection deferred or pending ingestion: {e}")
    collection = None

# Initialize Dense Vector Embedder
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Request / Response Schemas
class QueryRequest(BaseModel):
    drug: str
    query: str
    audience: Optional[str] = "patient"

class SourceChunk(BaseModel):
    chunk_id: Optional[str] = None
    section: str
    page: int
    text: str

class VerificationResult(BaseModel):
    quotes_valid: bool
    numbers_valid: bool

class ReadabilityResult(BaseModel):
    flesch_kincaid_grade: float
    is_patient_accessible: Optional[bool] = True

class AskResponse(BaseModel):
    type: str  # "answer" | "emergency" | "error"
    message: Optional[str] = None
    answer: Optional[str] = None
    verification: Optional[VerificationResult] = None
    readability: Optional[ReadabilityResult] = None
    sources: Optional[List[SourceChunk]] = None
    emergency_info: Optional[dict] = None

# API Endpoints

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "Ritam Medical AI API",
        "vector_store_active": collection is not None
    }

@app.get("/drugs", tags=["Drugs"])
def list_drugs():
    return {
        "drugs": [
            {"id": "metformin", "name": "Metformin"},
            {"id": "acetaminophen", "name": "Acetaminophen (Paracetamol)"},
            {"id": "omeprazole", "name": "Omeprazole"},
            {"id": "amoxicillin", "name": "Amoxicillin"},
            {"id": "cetirizine", "name": "Cetirizine"},
            {"id": "levothyroxine", "name": "Levothyroxine Sodium"}
        ]
    }

@app.post("/ask", response_model=AskResponse, tags=["Query"])
def ask_question(request: QueryRequest):
    # 1. Safety Gate Check for Emergency Intent
    emergency_contact = check_emergency_intent(request.query)
    if emergency_contact:
        return AskResponse(
            type="emergency",
            message="Immediate Critical Health Emergency Detected. Please seek urgent medical assistance or contact emergency service lines immediately.",
            emergency_info=emergency_contact,
            sources=[]
        )

    # 2. Check Vector Database Availability
    if not collection:
        raise HTTPException(
            status_code=500,
            detail="Vector database store not found at data/chroma_db. Run backend/ingest.py to populate drug labels."
        )

    # 3. Vector Similarity Search Querying
    query_vector = embedding_model.encode([request.query]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=3,
        where={"drug": request.drug.lower()}
    )

    retrieved_chunks = []
    if results and results.get("documents") and results["documents"][0]:
        for idx, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][idx]
            retrieved_chunks.append({
                "chunk_id": results["ids"][0][idx],
                "text": doc_text,
                "section": meta.get("section", "General"),
                "page": meta.get("page", 1)
            })

    # Return structured fallback if no matching context chunks exist in vector storage
    if not retrieved_chunks:
        return AskResponse(
            type="answer",
            answer=f"The official FDA label for {request.drug.capitalize()} does not contain information matching your query.",
            sources=[],
            verification=VerificationResult(quotes_valid=True, numbers_valid=True),
            readability=ReadabilityResult(flesch_kincaid_grade=0.0, is_patient_accessible=True)
        )

    # 4. Gemini Answer Synthesis
    answer = generate_medical_answer(
        query=request.query,
        drug=request.drug,
        context_chunks=retrieved_chunks,
        audience=request.audience
    )

    # 5. Hallucination & Readability Verification
    quotes_valid = verify_grounded_quotes(answer, retrieved_chunks)
    numbers_valid = verify_numerical_consistency(answer, retrieved_chunks)
    readability = calculate_readability(answer)

    formatted_sources = [
        SourceChunk(
            chunk_id=c.get("chunk_id"),
            section=c["section"],
            page=c["page"],
            text=c["text"]
        ) for c in retrieved_chunks
    ]

    return AskResponse(
        type="answer",
        answer=answer,
        verification=VerificationResult(
            quotes_valid=quotes_valid,
            numbers_valid=numbers_valid
        ),
        readability=ReadabilityResult(
            flesch_kincaid_grade=readability.get("flesch_kincaid_grade", 0.0),
            is_patient_accessible=readability.get("is_patient_accessible", True)
        ),
        sources=formatted_sources
    )