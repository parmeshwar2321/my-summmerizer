"""
FastAPI route handlers for On-Demand Document Summarization.
"""

import logging
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from api.schemas import DocumentSummaryRequest, DocumentSummaryResponse
from pipelines.doc_summarizer import DocumentSummarizer

logger = logging.getLogger("DocRoutes")
router = APIRouter(tags=["Document Summarization"])


@router.post(
    "/summarize-doc",
    response_model=DocumentSummaryResponse,
    summary="Summarize raw text document using Map-Reduce"
)
async def summarize_doc_endpoint(payload: DocumentSummaryRequest, request: Request):
    """
    On-demand document summarization using LangChain Map-Reduce chunking.
    Handles documents exceeding model context window length.
    """
    summarizer: DocumentSummarizer = getattr(request.app.state, "doc_summarizer", None)
    if not summarizer:
        raise HTTPException(status_code=503, detail="Document summarizer is not initialized.")

    try:
        result = summarizer.summarize(payload.text)
        return DocumentSummaryResponse(
            status="success",
            title=payload.title,
            executive_summary=result.executive_summary,
            key_findings=result.key_findings,
            chunk_summaries=result.chunk_summaries,
            total_chunks=result.total_chunks,
            original_length_chars=result.original_length_chars
        )
    except Exception as e:
        logger.error(f"Error during document summarization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post(
    "/summarize-doc/upload",
    response_model=DocumentSummaryResponse,
    summary="Summarize uploaded document file (.txt, .md)"
)
async def summarize_file_upload_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Document file to summarize (.txt, .md, etc.)"),
    title: str = Form(None)
):
    """
    On-demand document summarization triggered by file upload.
    Reads file content and runs Map-Reduce pipeline.
    """
    summarizer: DocumentSummarizer = getattr(request.app.state, "doc_summarizer", None)
    if not summarizer:
        raise HTTPException(status_code=503, detail="Document summarizer is not initialized.")

    try:
        content_bytes = await file.read()
        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content_bytes.decode("latin-1", errors="replace")

        doc_title = title or file.filename or "Uploaded Document"
        result = summarizer.summarize(text_content)

        return DocumentSummaryResponse(
            status="success",
            title=doc_title,
            executive_summary=result.executive_summary,
            key_findings=result.key_findings,
            chunk_summaries=result.chunk_summaries,
            total_chunks=result.total_chunks,
            original_length_chars=result.original_length_chars
        )
    except Exception as e:
        logger.error(f"Error processing uploaded file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File summarization failed: {str(e)}")
