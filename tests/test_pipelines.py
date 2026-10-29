"""
Tests for LangChain Email Pipeline and Document Summarizer.
"""

import pytest
import asyncio
from llm.model_loader import MockLangChainWrapper
from pipelines.email_pipeline import (
    EmailPipeline,
    EmailAnalysisOutput,
    extract_json_from_text,
    AlwaysOnEmailWatcher,
)
from pipelines.doc_summarizer import DocumentSummarizer, DocumentSummaryOutput


def test_extract_json_from_text():
    # 1. Clean JSON
    raw1 = '{"summary": "Test summary", "action_items": ["item1"], "urgency": "High", "sentiment": "Positive"}'
    res1 = extract_json_from_text(raw1)
    assert res1["summary"] == "Test summary"
    assert res1["urgency"] == "High"

    # 2. Markdown codeblock
    raw2 = '```json\n{"summary": "Codeblock summary", "action_items": [], "urgency": "Low", "sentiment": "Neutral"}\n```'
    res2 = extract_json_from_text(raw2)
    assert res2["summary"] == "Codeblock summary"
    assert res2["urgency"] == "Low"

    # 3. Preamble and postamble
    raw3 = 'Here is the output:\n{"summary": "Noisy summary", "action_items": ["do it"], "urgency": "Medium", "sentiment": "Negative"}\nHope this helps!'
    res3 = extract_json_from_text(raw3)
    assert res3["summary"] == "Noisy summary"
    assert res3["action_items"] == ["do it"]


def test_email_pipeline_analysis():
    mock_llm = MockLangChainWrapper()
    pipeline = EmailPipeline(llm=mock_llm)

    result = pipeline.analyze(
        content="Urgent issue with database latency",
        sender="dba@company.com",
        subject="DB Latency"
    )

    assert isinstance(result, EmailAnalysisOutput)
    assert len(result.summary) > 0
    assert isinstance(result.action_items, list)
    assert result.urgency in ["Low", "Medium", "High", "Critical"]
    assert result.sentiment in ["Positive", "Neutral", "Negative"]


@pytest.mark.asyncio
async def test_always_on_email_watcher():
    mock_llm = MockLangChainWrapper()
    pipeline = EmailPipeline(llm=mock_llm)

    analyzed_results = []

    async def callback(res):
        analyzed_results.append(res)

    watcher = AlwaysOnEmailWatcher(pipeline=pipeline, on_analyzed_callback=callback)
    await watcher.start()

    await watcher.push_email({
        "sender": "alerts@ops.com",
        "subject": "System Warning",
        "content": "Check disk usage immediately."
    })

    # Give the queue brief time to process
    await asyncio.sleep(0.1)
    await watcher.stop()

    assert len(analyzed_results) == 1
    assert isinstance(analyzed_results[0], EmailAnalysisOutput)


def test_document_summarizer_single_chunk():
    mock_llm = MockLangChainWrapper()
    summarizer = DocumentSummarizer(llm=mock_llm, chunk_size=1000)

    short_doc = "This is a short document that easily fits into a single chunk."
    result = summarizer.summarize(short_doc)

    assert isinstance(result, DocumentSummaryOutput)
    assert result.total_chunks == 1
    assert len(result.executive_summary) > 0


def test_document_summarizer_map_reduce():
    mock_llm = MockLangChainWrapper()
    # Use very small chunk size to trigger multi-chunk Map-Reduce
    summarizer = DocumentSummarizer(llm=mock_llm, chunk_size=80, chunk_overlap=10)

    long_doc = (
        "Introduction to Small Language Models. "
        "Small Language Models (SLMs) offer tremendous efficiency. "
        "They run on commodity hardware and consumer devices with minimal latency. "
        "Fine-tuning with LoRA allows high task-specific accuracy."
    )

    result = summarizer.summarize(long_doc)

    assert isinstance(result, DocumentSummaryOutput)
    assert result.total_chunks > 1
    assert len(result.chunk_summaries) == result.total_chunks
    assert len(result.executive_summary) > 0
    assert len(result.key_findings) > 0
