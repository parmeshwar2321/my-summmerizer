"""
LangChain Pipelines for Email Intelligence and Document Summarization.
"""

from pipelines.email_pipeline import EmailPipeline, EmailAnalysisOutput
from pipelines.doc_summarizer import DocumentSummarizer, DocumentSummaryOutput

__all__ = [
    "EmailPipeline",
    "EmailAnalysisOutput",
    "DocumentSummarizer",
    "DocumentSummaryOutput",
]
