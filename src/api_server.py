"""
FastAPI server for the Combined Fact-Checking System.

Endpoints:
  POST /verify       - Verify a single claim
  POST /verify/batch - Verify multiple claims
  GET  /health       - Health check & loaded model info

Run:
  uvicorn src.api_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import mimetypes
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from models.inference import FactCheckPipeline

# Global pipeline instance
pipeline: Optional[FactCheckPipeline] = None

# Force correct MIME type for JS modules (Windows fix)
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def _parse_cors_origins() -> List[str]:
    """Parse CORS origins from env var. Defaults to wildcard."""
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _normalize_claim(text: str) -> str:
    """Normalize and validate claim text."""
    normalized = text.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="Claim must not be empty")
    return normalized


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup."""
    global pipeline
    print("Loading Fact-Checking Pipeline...")
    pipeline = FactCheckPipeline()
    pipeline.load()
    print("Pipeline ready.")
    yield
    print("Shutting down pipeline.")


app = FastAPI(
    title="Fact-Checking API",
    description="Combined LIAR + FEVER + FakeNewsNet fact-checking system",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Schemas ---

class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="The claim text to verify")
    top_k_evidence: int = Field(3, ge=1, le=10, description="Number of evidence documents to retrieve")


class BatchVerifyRequest(BaseModel):
    claims: List[str] = Field(..., min_items=1, max_items=50, description="List of claims to verify")
    top_k_evidence: int = Field(3, ge=1, le=10)


class EvidenceItem(BaseModel):
    doc_id: str
    score: float
    snippet: str


class ModelDetail(BaseModel):
    credibility_score: float
    predicted_class: int
    predicted_label: Optional[str] = None
    probabilities: List[float]
    temperature: float
    weight: float


class VerifyResponse(BaseModel):
    claim: str
    credibility_score: float
    verdict: str
    evidence: List[EvidenceItem]
    model_details: dict
    inference_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    active_models: List[str]
    retrieval_docs: int


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and loaded models."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return HealthResponse(
        status="ok",
        active_models=list(pipeline.models.keys()),
        retrieval_docs=len(pipeline.retriever.corpus) if pipeline.retriever else 0,
    )


def _to_verify_response(result: dict) -> VerifyResponse:
    return VerifyResponse(
        claim=result["claim"],
        credibility_score=result["credibility_score"],
        verdict=result["verdict"],
        evidence=[EvidenceItem(**e) for e in result["evidence"]],
        model_details=result["model_details"],
        inference_time_ms=result.get("inference_time_ms")
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify_claim(req: VerifyRequest):
    """Verify a single claim and return verdict with evidence."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    claim = _normalize_claim(req.claim)

    try:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(pipeline.verify, claim, req.top_k_evidence)
        elapsed = time.perf_counter() - t0
        result["inference_time_ms"] = round(elapsed * 1000, 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc

    return _to_verify_response(result)


@app.post("/verify/batch", response_model=List[VerifyResponse])
async def verify_batch(req: BatchVerifyRequest):
    """Verify multiple claims using thread pool for non-blocking execution."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    normalized_claims = [_normalize_claim(claim) for claim in req.claims]

    def _run_batch():
        batch_results = []
        for claim in normalized_claims:
            result = pipeline.verify(claim, top_k_evidence=req.top_k_evidence)
            batch_results.append(result)
        return batch_results

    try:
        raw_results = await asyncio.to_thread(_run_batch)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch verification failed: {exc}") from exc

    return [_to_verify_response(r) for r in raw_results]


# --- Static Files (MOUNT LAST) ---

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".js"):
            response.headers["content-type"] = "application/javascript"
        return response

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')

app.mount("/", SPAStaticFiles(directory="frontend"), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)
