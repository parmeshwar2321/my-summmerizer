"""
FastAPI Application Entrypoint for SLM Serving.
Manages model lifecycle, background watcher tasks, and exposes pipeline endpoints.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.email_routes import router as email_router
from api.routes.doc_routes import router as doc_router
from api.schemas import HealthResponse
from llm.model_loader import SLMModelLoader
from llm.config import ModelConfig
from pipelines.email_pipeline import EmailPipeline, AlwaysOnEmailWatcher
from pipelines.doc_summarizer import DocumentSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SLMServer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown events:
    1. Loads the SLM into memory once.
    2. Initializes LangChain email and document pipelines.
    3. Starts the always-on background email watcher.
    """
    logger.info("Initializing SLM Serving Application...")

    # Load Model
    model_config = ModelConfig()
    loader = SLMModelLoader.get_instance(model_config)
    llm = loader.load()

    # Initialize LangChain pipelines
    email_pipeline = EmailPipeline(llm=llm)
    doc_summarizer = DocumentSummarizer(llm=llm)

    # Start Always-On Email Watcher
    async def on_email_analyzed(result):
        logger.info(f"[Always-On Watcher] Analyzed incoming email: Urgency={result.urgency}, Summary={result.summary[:60]}...")

    email_watcher = AlwaysOnEmailWatcher(pipeline=email_pipeline, on_analyzed_callback=on_email_analyzed)
    await email_watcher.start()

    # Attach instances to app.state
    app.state.model_loader = loader
    app.state.llm = llm
    app.state.email_pipeline = email_pipeline
    app.state.doc_summarizer = doc_summarizer
    app.state.email_watcher = email_watcher

    logger.info("All pipelines initialized and ready to serve requests.")

    yield

    # Teardown
    logger.info("Shutting down pipelines and background watchers...")
    await email_watcher.stop()
    logger.info("Shutdown completed.")


app = FastAPI(
    title="SLM Intelligence Platform",
    description=(
        "Production-grade serving backend for Small Language Models (SLMs) featuring:\n"
        "1. Always-on email analysis (summary, action items, urgency, sentiment).\n"
        "2. On-demand Map-Reduce document summarization for arbitrary-length files.\n"
        "3. High-throughput containerized serving."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(email_router)
app.include_router(doc_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns server health, loaded model ID, and execution device."""
    model_config = ModelConfig()
    is_mock = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")
    return HealthResponse(
        status="healthy",
        model=model_config.model_id_or_path,
        device=model_config.device,
        mock_mode=is_mock
    )


@app.get("/", tags=["Root"])
async def root():
    """Root landing page."""
    return {
        "message": "Small Language Model (SLM) API is running.",
        "docs_url": "/docs",
        "endpoints": [
            "POST /analyze-email",
            "POST /enqueue-email",
            "POST /summarize-doc",
            "POST /summarize-doc/upload",
            "GET /health"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
