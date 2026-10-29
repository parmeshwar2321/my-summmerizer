"""
FastAPI route handlers for Email Intelligence Pipeline.
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from api.schemas import EmailAnalysisRequest, EmailAnalysisResponse
from pipelines.email_pipeline import EmailPipeline, AlwaysOnEmailWatcher

logger = logging.getLogger("EmailRoutes")
router = APIRouter(tags=["Email Intelligence"])


@router.post(
    "/analyze-email",
    response_model=EmailAnalysisResponse,
    summary="Analyze incoming email synchronously"
)
async def analyze_email_endpoint(payload: EmailAnalysisRequest, request: Request):
    """
    Analyzes an email to extract summary, action items, urgency, and sentiment.
    """
    pipeline: EmailPipeline = getattr(request.app.state, "email_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Email pipeline is not initialized.")

    try:
        result = pipeline.analyze(
            content=payload.content,
            sender=payload.sender,
            subject=payload.subject
        )
        return EmailAnalysisResponse(
            status="success",
            summary=result.summary,
            action_items=result.action_items,
            urgency=result.urgency,
            sentiment=result.sentiment
        )
    except Exception as e:
        logger.error(f"Error during email analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post(
    "/enqueue-email",
    summary="Enqueue incoming email into always-on background listener"
)
async def enqueue_email_endpoint(payload: EmailAnalysisRequest, request: Request):
    """
    Enqueues an incoming email into the always-on background watcher queue.
    """
    watcher: AlwaysOnEmailWatcher = getattr(request.app.state, "email_watcher", None)
    if not watcher:
        raise HTTPException(status_code=503, detail="Always-on watcher is not running.")

    await watcher.push_email({
        "sender": payload.sender,
        "subject": payload.subject,
        "content": payload.content
    })
    return {"status": "enqueued", "message": "Email successfully enqueued for background processing."}
