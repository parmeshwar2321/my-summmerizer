"""
On-Demand Document Summarizer using LangChain Map-Reduce Chunking.
Handles arbitrary-length documents exceeding the Small Language Model's context window.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

logger = logging.getLogger("DocumentSummarizer")


class DocumentSummaryOutput(BaseModel):
    """Structured response schema for document summarization."""
    executive_summary: str = Field(description="Synthesized high-level executive summary.")
    key_findings: List[str] = Field(default_factory=list, description="Key points extracted across the entire document.")
    chunk_summaries: List[str] = Field(default_factory=list, description="Intermediate summaries from the Map stage.")
    total_chunks: int = Field(description="Total chunks processed by the map step.")
    original_length_chars: int = Field(description="Character count of the uploaded document.")


# Map Prompt: processes individual chunk
MAP_PROMPT_TEMPLATE = """<|im_start|>system
You are a precise document analyst. Summarize the following excerpt concisely.
Highlight key points, metrics, and core decisions. Keep the summary under 100 words.
<|im_end|>
<|im_start|>user
Document Section {chunk_index} of {total_chunks}:
{chunk_text}
<|im_end|>
<|im_start|>assistant
"""

# Reduce Prompt: synthesizes all intermediate chunk summaries into cohesive document summary
REDUCE_PROMPT_TEMPLATE = """<|im_start|>system
You are an executive assistant. The following are summaries of consecutive sections from a long document.
Synthesize these section summaries into:
1. A comprehensive, coherent executive summary.
2. A bulleted list of the top 3-5 overall key takeaways.
Format clearly.
<|im_end|>
<|im_start|>user
Section Summaries:
{combined_summaries}
<|im_end|>
<|im_start|>assistant
"""


class DocumentSummarizer:
    """
    On-demand document summarizer using Map-Reduce chunking.
    Triggers only when invoked (e.g. on file or text upload).
    """

    def __init__(
        self,
        llm: LLM,
        chunk_size: int = 1500,
        chunk_overlap: int = 150
    ):
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.map_prompt = PromptTemplate(
            template=MAP_PROMPT_TEMPLATE,
            input_variables=["chunk_text", "chunk_index", "total_chunks"]
        )
        self.reduce_prompt = PromptTemplate(
            template=REDUCE_PROMPT_TEMPLATE,
            input_variables=["combined_summaries"]
        )

        self.map_chain = self.map_prompt | self.llm
        self.reduce_chain = self.reduce_prompt | self.llm

    def summarize(self, text: str, max_reduce_tokens: Optional[int] = None) -> DocumentSummaryOutput:
        """
        Executes Map-Reduce summarization on input document text.
        """
        original_length = len(text)
        if not text.strip():
            return DocumentSummaryOutput(
                executive_summary="Empty document provided.",
                key_findings=[],
                chunk_summaries=[],
                total_chunks=0,
                original_length_chars=0
            )

        # 1. Chunking
        chunks = self.text_splitter.split_text(text)
        total_chunks = len(chunks)
        logger.info(f"Split document ({original_length} chars) into {total_chunks} chunks.")

        # Single chunk bypass for low latency
        if total_chunks == 1:
            logger.info("Document fits within single chunk; executing direct summarization...")
            map_res = self.map_chain.invoke({
                "chunk_text": chunks[0],
                "chunk_index": 1,
                "total_chunks": 1
            })
            clean_res = map_res.strip()
            return DocumentSummaryOutput(
                executive_summary=clean_res,
                key_findings=[clean_res],
                chunk_summaries=[clean_res],
                total_chunks=1,
                original_length_chars=original_length
            )

        # 2. Map Stage: Summarize each chunk
        logger.info(f"Executing MAP phase across {total_chunks} chunks...")
        chunk_summaries: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {idx}/{total_chunks}...")
            res = self.map_chain.invoke({
                "chunk_text": chunk,
                "chunk_index": idx,
                "total_chunks": total_chunks
            })
            chunk_summaries.append(res.strip())

        # 3. Reduce Stage: Synthesize all chunk summaries
        logger.info(f"Executing REDUCE phase on {len(chunk_summaries)} chunk summaries...")
        combined_summaries = "\n\n".join(
            [f"[Section {i+1}]: {summary}" for i, summary in enumerate(chunk_summaries)]
        )

        # Hierarchical reduce check if combined summaries exceed chunk budget
        if len(combined_summaries) > (self.chunk_size * 2):
            logger.info("Intermediate summaries exceed single prompt window; performing secondary reduce...")
            intermediate_chunks = self.text_splitter.split_text(combined_summaries)
            intermediate_summaries = []
            for ic in intermediate_chunks:
                res = self.map_chain.invoke({
                    "chunk_text": ic,
                    "chunk_index": 1,
                    "total_chunks": len(intermediate_chunks)
                })
                intermediate_summaries.append(res.strip())
            combined_summaries = "\n\n".join(intermediate_summaries)

        final_reduced = self.reduce_chain.invoke({
            "combined_summaries": combined_summaries
        }).strip()

        # Extract key findings from bullet points if present
        key_findings = []
        for line in final_reduced.split("\n"):
            line = line.strip()
            if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
                clean_finding = line.lstrip("-*•0123456789. ")
                if clean_finding:
                    key_findings.append(clean_finding)

        if not key_findings:
            key_findings = chunk_summaries[:5]

        return DocumentSummaryOutput(
            executive_summary=final_reduced,
            key_findings=key_findings,
            chunk_summaries=chunk_summaries,
            total_chunks=total_chunks,
            original_length_chars=original_length
        )
