"""
Pydantic schemas for FastAPI request and response payloads.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EmailAnalysisRequest(BaseModel):
    """Payload for POST /analyze-email endpoint."""
    content: str = Field(..., description="The main text body of the incoming email.", min_length=1)
    sender: str = Field(default="Unknown", description="Email address or name of the sender.")
    subject: str = Field(default="(No Subject)", description="Subject line of the email.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sender": "lead-engineer@platform.org",
                "subject": "ACTION REQUIRED: Production Redis Cluster Memory Critical",
                "content": "Redis memory consumption has exceeded 92%. Please increase node allocation and review eviction policies by 2 PM."
            }
        }
    }


class EmailAnalysisResponse(BaseModel):
    """Response payload for email analysis."""
    status: str = Field(default="success")
    summary: str = Field(description="Synthesized summary of the email.")
    action_items: List[str] = Field(description="List of extracted actionable items.")
    urgency: Literal["Low", "Medium", "High", "Critical"] = Field(description="Urgency classification.")
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(description="Sentiment classification.")


class DocumentSummaryRequest(BaseModel):
    """Payload for POST /summarize-doc endpoint (JSON text body)."""
    text: str = Field(..., description="Document content to summarize.", min_length=1)
    title: Optional[str] = Field(default="Uploaded Document", description="Optional title or file name.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Quarterly Technical Report",
                "text": "Over the last quarter, we successfully optimized our machine learning inference pipeline... (long text)"
            }
        }
    }


class DocumentSummaryResponse(BaseModel):
    """Response payload for document summarization."""
    status: str = Field(default="success")
    title: Optional[str] = None
    executive_summary: str = Field(description="Synthesized Map-Reduce executive summary.")
    key_findings: List[str] = Field(description="Top key takeaways.")
    chunk_summaries: List[str] = Field(description="Intermediate chunk summaries.")
    total_chunks: int = Field(description="Number of chunks processed.")
    original_length_chars: int = Field(description="Original character count.")


class HealthResponse(BaseModel):
    """Response payload for GET /health."""
    status: str
    model: str
    device: str
    mock_mode: bool
