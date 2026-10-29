"""
FastAPI Serving Layer for SLM Pipelines.
"""

from api.schemas import (
    EmailAnalysisRequest,
    EmailAnalysisResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    HealthResponse,
)

__all__ = [
    "EmailAnalysisRequest",
    "EmailAnalysisResponse",
    "DocumentSummaryRequest",
    "DocumentSummaryResponse",
    "HealthResponse",
]
