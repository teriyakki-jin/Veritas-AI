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
from statistics import mean
from contextlib import asynccontextmanager
from pathlib import Path
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
from models.fusion import score_to_verdict
from utils.article_scraper import scrape_article
from utils.claim_extractor import extract_claims

# Global pipeline instance
pipeline: Optional[FactCheckPipeline] = None

# Force correct MIME type for JS modules (Windows fix)
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def _load_dotenv() -> None:
    """Load .env from project root without extra dependencies."""
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


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
    version="1.1.1",
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


class AnalyzeArticleRequest(BaseModel):
    url: Optional[str] = Field(None, description="Article URL (http/https)")
    article_text: Optional[str] = Field(None, description="Raw article text if URL scraping is not used")
    top_k_evidence: int = Field(3, ge=1, le=10)
    max_claims: int = Field(5, ge=1, le=15)


class ArticleSource(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    text_length: int
    text_preview: str


class AnalyzeArticleResponse(BaseModel):
    source: ArticleSource
    extracted_claims: List[str]
    claim_results: List[VerifyResponse]
    article_credibility_score: float
    article_verdict: str
    extraction_time_ms: float
    verification_time_ms: float


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
        inference_time_ms=result.get("inference_time_ms"),
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


@app.post("/analyze/article", response_model=AnalyzeArticleResponse)
async def analyze_article(req: AnalyzeArticleRequest):
    """Scrape (or accept text), extract claims, and verify each claim."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    has_url = bool(req.url and req.url.strip())
    has_text = bool(req.article_text and req.article_text.strip())
    if not has_url and not has_text:
        raise HTTPException(status_code=422, detail="Either 'url' or 'article_text' must be provided")

    extraction_t0 = time.perf_counter()
    title = None
    source_url = req.url.strip() if has_url else None

    try:
        if has_url:
            scraped = await asyncio.to_thread(scrape_article, req.url.strip())
            article_text = scraped.text
            title = scraped.title
            source_url = scraped.url
        else:
            article_text = req.article_text.strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Article extraction failed: {exc}") from exc

    claims = extract_claims(article_text, max_claims=req.max_claims)
    extraction_elapsed = (time.perf_counter() - extraction_t0) * 1000

    if not claims:
        raise HTTPException(status_code=422, detail="No verifiable claims could be extracted from article")

    verify_t0 = time.perf_counter()

    def _verify_claims():
        results = []
        for claim in claims:
            t0 = time.perf_counter()
            result = pipeline.verify(claim, top_k_evidence=req.top_k_evidence)
            result["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            results.append(result)
        return results

    try:
        raw_results = await asyncio.to_thread(_verify_claims)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Claim verification failed: {exc}") from exc

    verification_elapsed = (time.perf_counter() - verify_t0) * 1000
    scores = [r["credibility_score"] for r in raw_results]
    article_score = round(float(mean(scores)), 4) if scores else 0.5
    article_verdict = score_to_verdict(article_score)

    response_results = [_to_verify_response(r) for r in raw_results]
    preview = article_text[:280]

    return AnalyzeArticleResponse(
        source=ArticleSource(
            url=source_url,
            title=title,
            text_length=len(article_text),
            text_preview=preview,
        ),
        extracted_claims=claims,
        claim_results=response_results,
        article_credibility_score=article_score,
        article_verdict=article_verdict,
        extraction_time_ms=round(extraction_elapsed, 1),
        verification_time_ms=round(verification_elapsed, 1),
    )


# --- Static Files (MOUNT LAST) ---

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".js"):
            response.headers["content-type"] = "application/javascript"
        return response


@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


app.mount("/", SPAStaticFiles(directory="frontend"), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)
