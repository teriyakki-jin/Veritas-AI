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
import json
import logging
import mimetypes
import os
import sys
import time
from collections import defaultdict
from statistics import mean
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from models.inference import FactCheckPipeline
from models.fusion import score_to_verdict
from utils.openai_client import OpenAIClient
from utils.article_scraper import scrape_article
from utils.claim_extractor import extract_claims

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


# Per-path rate limits: (max_calls, window_seconds)
_RATE_LIMITS: Dict[str, tuple] = {
    "/verify/batch":    (10, 60),
    "/analyze/article": (5,  60),
    "/verify":          (30, 60),
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-IP rate limiter (no external dependencies)."""

    def __init__(self, app):
        super().__init__(app)
        self._history: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        limit_cfg = None
        for prefix, cfg in _RATE_LIMITS.items():
            if path.startswith(prefix):
                limit_cfg = cfg
                break

        if limit_cfg is not None:
            max_calls, window = limit_cfg
            client_ip = (request.client.host if request.client else "unknown")
            key = f"{client_ip}:{path}"
            now = time.monotonic()
            cutoff = now - window

            # Prune stale timestamps
            self._history[key] = [t for t in self._history[key] if t > cutoff]

            if len(self._history[key]) >= max_calls:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please try again later."},
                )

            self._history[key].append(now)

        return await call_next(request)


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
    logger.info("Loading Fact-Checking Pipeline...")
    app.state.pipeline = FactCheckPipeline()
    app.state.pipeline.load()
    try:
        app.state.openai_client = OpenAIClient()
        if app.state.openai_client.available:
            logger.info("OpenAI client enabled (model=%s).", app.state.openai_client.model)
        else:
            logger.info("OpenAI client disabled (set OPENAI_ENABLED=true and OPENAI_API_KEY).")
    except Exception:
        logger.exception("OpenAI client initialization failed; continuing without LLM assist.")
        app.state.openai_client = None
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down pipeline.")


app = FastAPI(
    title="Fact-Checking API",
    description="Combined LIAR + FEVER + FakeNewsNet fact-checking system",
    version="1.1.1",
    lifespan=lifespan,
)

_cors_origins = _parse_cors_origins()
_cors_allow_credentials = "*" not in _cors_origins

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Schemas ---

class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="The claim text to verify")
    top_k_evidence: int = Field(3, ge=1, le=10, description="Number of evidence documents to retrieve")


class BatchVerifyRequest(BaseModel):
    claims: List[str] = Field(..., min_length=1, max_length=50, description="List of claims to verify")
    top_k_evidence: int = Field(3, ge=1, le=10)


class EvidenceItem(BaseModel):
    doc_id: str
    score: float
    snippet: str


class EvidenceItemExplained(BaseModel):
    doc_id: str
    score: float
    snippet: str
    contribution: float
    contribution_label: str  # "supports" | "refutes" | "neutral"


class ExplainResponse(BaseModel):
    claim: str
    credibility_score: float
    verdict: str
    evidence: List[EvidenceItemExplained]
    model_details: dict
    inference_time_ms: Optional[float] = None


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


class VerifyAssistRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="The claim text to verify")
    top_k_evidence: int = Field(3, ge=1, le=10)
    include_model_result: bool = Field(True, description="Include local model result in OpenAI prompt")
    require_openai: bool = Field(False, description="If true, return 503 when OpenAI is not configured")


class VerifyAssistResponse(BaseModel):
    local_result: VerifyResponse
    llm_used: bool
    llm_analysis: Optional[dict] = None


# --- Endpoints ---

def _get_pipeline(request: Request) -> FactCheckPipeline:
    p = getattr(request.app.state, "pipeline", None)
    if p is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return p


def _get_openai_client(request: Request) -> Optional[OpenAIClient]:
    return getattr(request.app.state, "openai_client", None)


@app.get("/health", response_model=HealthResponse)
async def health_check(pipeline: FactCheckPipeline = Depends(_get_pipeline)):
    """Check system health and loaded models."""
    return HealthResponse(
        status="ok",
        active_models=list(pipeline.models.keys()),
        retrieval_docs=len(pipeline.retriever.corpus) if pipeline.retriever else 0,
    )


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/verify/stream")
async def verify_stream(
    claim: str = Query(..., min_length=1, description="Claim text to verify"),
    top_k_evidence: int = Query(3, ge=1, le=10),
    pipeline: FactCheckPipeline = Depends(_get_pipeline),
):
    """Stream verification progress via Server-Sent Events.

    Events emitted in order:
      retrieving  → BM25 search started
      evidence    → evidence docs retrieved
      verifying   → per-model inference started (one per model)
      model_done  → per-model result (one per model)
      fusing      → fusion started
      result      → final VerifyResponse payload
      done        → stream complete
      error       → unrecoverable error
    """
    claim = _normalize_claim(claim)

    async def generate():
        try:
            yield _sse("retrieving", {"message": "Retrieving evidence..."})

            evidence = await asyncio.to_thread(pipeline.retrieve_evidence, claim, top_k_evidence)
            ev_list = [
                {"doc_id": e["doc_id"], "score": e["score"], "snippet": e["text"]}
                for e in evidence
            ]
            yield _sse("evidence", {"count": len(ev_list), "evidence": ev_list})

            evidence_texts = [e["text"] for e in evidence]
            model_outputs = {}

            for model_name in pipeline.models:
                yield _sse("verifying", {"model": model_name})
                if model_name == "fever":
                    inp = claim + " [SEP] " + " [SEP] ".join(evidence_texts[:3]) if evidence_texts else claim
                else:
                    inp = claim
                logits = await asyncio.to_thread(pipeline.predict_single, model_name, inp)
                if logits.size > 0:
                    model_outputs[model_name] = logits
                yield _sse("model_done", {"model": model_name})

            yield _sse("fusing", {"message": "Computing final verdict..."})

            if model_outputs and pipeline.fusion:
                fusion_result = pipeline.fusion.fuse(model_outputs)
            else:
                fusion_result = {"credibility_score": 0.5, "verdict": "UNKNOWN (no models)", "model_details": {}}

            fusion_result = pipeline._enrich_model_details(fusion_result)
            result = {
                "claim": claim,
                "evidence": ev_list,
                "credibility_score": fusion_result["credibility_score"],
                "verdict": fusion_result["verdict"],
                "model_details": fusion_result["model_details"],
            }
            pipeline._cache.put(claim, top_k_evidence, result)

            yield _sse("result", result)
            yield _sse("done", {})

        except HTTPException as exc:
            yield _sse("error", {"message": exc.detail})
        except Exception:
            logger.exception("Stream verification error")
            yield _sse("error", {"message": "Verification failed. Please try again."})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
async def verify_claim(req: VerifyRequest, pipeline: FactCheckPipeline = Depends(_get_pipeline)):
    """Verify a single claim and return verdict with evidence."""
    claim = _normalize_claim(req.claim)

    try:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(pipeline.verify, claim, req.top_k_evidence)
        elapsed = time.perf_counter() - t0
        result = {**result, "inference_time_ms": round(elapsed * 1000, 1)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Claim verification error")
        raise HTTPException(status_code=500, detail="Verification failed. Please try again.") from exc

    return _to_verify_response(result)


@app.post("/verify/explain", response_model=ExplainResponse)
async def verify_explain(req: VerifyRequest, pipeline: FactCheckPipeline = Depends(_get_pipeline)):
    """Verify a claim and return per-evidence contribution scores (Leave-One-Out)."""
    claim = _normalize_claim(req.claim)

    try:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(pipeline.verify, claim, req.top_k_evidence)
        contributions = await asyncio.to_thread(
            pipeline.explain_evidence, claim, result["evidence"], req.top_k_evidence
        )
        elapsed = time.perf_counter() - t0
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Explain endpoint error")
        raise HTTPException(status_code=500, detail="Verification failed. Please try again.") from exc

    evidence_explained = [
        EvidenceItemExplained(**ev, **contrib)
        for ev, contrib in zip(result["evidence"], contributions)
    ]

    return ExplainResponse(
        claim=result["claim"],
        credibility_score=result["credibility_score"],
        verdict=result["verdict"],
        evidence=evidence_explained,
        model_details=result["model_details"],
        inference_time_ms=round(elapsed * 1000, 1),
    )


@app.post("/verify/assist", response_model=VerifyAssistResponse)
async def verify_claim_assist(
    req: VerifyAssistRequest,
    pipeline: FactCheckPipeline = Depends(_get_pipeline),
    openai_client: Optional[OpenAIClient] = Depends(_get_openai_client),
):
    claim = _normalize_claim(req.claim)
    try:
        t0 = time.perf_counter()
        local = await asyncio.to_thread(pipeline.verify, claim, req.top_k_evidence)
        local = {**local, "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as exc:
        logger.exception("verify/assist local verification error")
        raise HTTPException(status_code=500, detail="Verification failed. Please try again.") from exc

    if not openai_client or not openai_client.available:
        if req.require_openai:
            raise HTTPException(status_code=503, detail="OpenAI is not configured")
        return VerifyAssistResponse(local_result=_to_verify_response(local), llm_used=False, llm_analysis=None)

    llm_input = local if req.include_model_result else None
    try:
        llm_analysis = await asyncio.to_thread(openai_client.analyze_claim, claim, llm_input)
    except Exception as exc:
        logger.exception("verify/assist OpenAI call failed")
        if req.require_openai:
            raise HTTPException(status_code=502, detail="OpenAI call failed") from exc
        return VerifyAssistResponse(local_result=_to_verify_response(local), llm_used=False, llm_analysis=None)

    return VerifyAssistResponse(local_result=_to_verify_response(local), llm_used=True, llm_analysis=llm_analysis)


@app.post("/verify/batch", response_model=List[VerifyResponse])
async def verify_batch(req: BatchVerifyRequest, pipeline: FactCheckPipeline = Depends(_get_pipeline)):
    """Verify multiple claims using thread pool for non-blocking execution."""
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
        logger.exception("Batch verification error")
        raise HTTPException(status_code=500, detail="Batch verification failed. Please try again.") from exc

    return [_to_verify_response(r) for r in raw_results]


async def _fetch_article_text(req: AnalyzeArticleRequest) -> tuple[str, str | None, str | None]:
    """Returns (article_text, title, source_url)."""
    has_url = bool(req.url and req.url.strip())
    has_text = bool(req.article_text and req.article_text.strip())
    if not has_url and not has_text:
        raise HTTPException(status_code=422, detail="Either 'url' or 'article_text' must be provided")

    title: str | None = None
    source_url: str | None = req.url.strip() if has_url else None
    try:
        if has_url:
            scraped = await asyncio.to_thread(scrape_article, req.url.strip())
            article_text = scraped.text
            title = scraped.title
            source_url = scraped.url
        else:
            article_text = req.article_text.strip()
        return article_text, title, source_url
    except ValueError as exc:
        logger.warning("Article extraction failed: %s", exc)
        raise HTTPException(status_code=400, detail="Article extraction failed. Check the URL and try again.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during article extraction")
        raise HTTPException(status_code=500, detail="Article extraction failed. Please try again.") from exc


def _verify_all_claims(claims: List[str], top_k: int, pipeline: FactCheckPipeline) -> List[dict]:
    """Runs pipeline.verify for each claim synchronously."""
    results = []
    for claim in claims:
        t0 = time.perf_counter()
        result = pipeline.verify(claim, top_k_evidence=top_k)
        result = {**result, "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1)}
        results.append(result)
    return results


def _compute_article_verdict(results: List[dict]) -> tuple[float, str]:
    """Returns (article_credibility_score, article_verdict)."""
    scores = [r["credibility_score"] for r in results]
    article_score = round(float(mean(scores)), 4) if scores else 0.5
    return article_score, score_to_verdict(article_score)


@app.post("/analyze/article", response_model=AnalyzeArticleResponse)
async def analyze_article(req: AnalyzeArticleRequest, pipeline: FactCheckPipeline = Depends(_get_pipeline)):
    """Scrape (or accept text), extract claims, and verify each claim."""
    extraction_t0 = time.perf_counter()
    article_text, title, source_url = await _fetch_article_text(req)
    claims = extract_claims(article_text, max_claims=req.max_claims)
    extraction_elapsed = (time.perf_counter() - extraction_t0) * 1000

    if not claims:
        raise HTTPException(status_code=422, detail="No verifiable claims could be extracted from article")

    verify_t0 = time.perf_counter()

    try:
        raw_results = await asyncio.to_thread(_verify_all_claims, claims, req.top_k_evidence, pipeline)
    except Exception as exc:
        logger.exception("Article claim verification error")
        raise HTTPException(status_code=500, detail="Claim verification failed. Please try again.") from exc

    verification_elapsed = (time.perf_counter() - verify_t0) * 1000
    article_score, article_verdict = _compute_article_verdict(raw_results)

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

    _host = os.getenv("HOST", "127.0.0.1")
    _port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.api_server:app", host=_host, port=_port, reload=True)
