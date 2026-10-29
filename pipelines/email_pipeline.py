"""
Always-On Email Analysis Pipeline using LangChain.
Extracts:
1. Concise executive summary
2. Action items
3. Urgency rating ('Low', 'Medium', 'High', 'Critical')
4. Sentiment analysis ('Positive', 'Neutral', 'Negative')
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.llms import LLM

logger = logging.getLogger("EmailPipeline")


class EmailAnalysisOutput(BaseModel):
    """Structured analysis response schema for emails."""
    summary: str = Field(description="A concise 1-3 sentence summary of the email's core message.")
    action_items: List[str] = Field(default_factory=list, description="Explicit action items or requested tasks.")
    urgency: Literal["Low", "Medium", "High", "Critical"] = Field(
        default="Medium",
        description="Urgency classification based on deadlines and severity."
    )
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(
        default="Neutral",
        description="Overall emotional tone and sentiment of the sender."
    )


EMAIL_PROMPT_TEMPLATE = """<|im_start|>system
You are an executive email intelligence AI. Analyze the following incoming email carefully.
Output your analysis STRICTLY as a single JSON object with no preamble, markdown ticks, or explanation.
The JSON object must follow this exact format:
{{
  "summary": "Concise 1-3 sentence summary",
  "action_items": ["Action 1", "Action 2"],
  "urgency": "Low" | "Medium" | "High" | "Critical",
  "sentiment": "Positive" | "Neutral" | "Negative"
}}
<|im_end|>
<|im_start|>user
Sender: {sender}
Subject: {subject}
Content:
{content}
<|im_end|>
<|im_start|>assistant
"""


def extract_json_from_text(text: str) -> dict:
    """Robustly extracts JSON from raw LLM output, handling markdown blocks and preambles."""
    # Attempt 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Attempt 2: code block extraction ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Attempt 3: first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = text[start:end+1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    # Fallback default if parsing completely fails
    logger.warning(f"Could not parse valid JSON from text: {text[:200]}... Returning fallback.")
    return {
        "summary": text.strip()[:300],
        "action_items": [],
        "urgency": "Medium",
        "sentiment": "Neutral"
    }


class EmailPipeline:
    """
    LangChain pipeline for analyzing incoming emails.
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=EmailAnalysisOutput)
        self.prompt = PromptTemplate(
            template=EMAIL_PROMPT_TEMPLATE,
            input_variables=["sender", "subject", "content"],
        )
        self.chain = self.prompt | self.llm

    def analyze(self, content: str, sender: str = "Unknown", subject: str = "(No Subject)") -> EmailAnalysisOutput:
        """
        Executes the LangChain analysis chain on an individual email.
        """
        logger.info(f"Processing email from '{sender}' with subject '{subject}'...")
        raw_response = self.chain.invoke({
            "sender": sender,
            "subject": subject,
            "content": content
        })

        if isinstance(raw_response, str):
            parsed_data = extract_json_from_text(raw_response)
        elif isinstance(raw_response, dict):
            parsed_data = raw_response
        else:
            parsed_data = extract_json_from_text(str(raw_response))

        # Validate through Pydantic
        try:
            return EmailAnalysisOutput(**parsed_data)
        except Exception as e:
            logger.error(f"Pydantic validation error: {e}. Coercing fields.")
            return EmailAnalysisOutput(
                summary=str(parsed_data.get("summary", "Analysis completed.")),
                action_items=list(parsed_data.get("action_items", [])),
                urgency=parsed_data.get("urgency", "Medium") if parsed_data.get("urgency") in ["Low", "Medium", "High", "Critical"] else "Medium",
                sentiment=parsed_data.get("sentiment", "Neutral") if parsed_data.get("sentiment") in ["Positive", "Neutral", "Negative"] else "Neutral"
            )


class AlwaysOnEmailWatcher:
    """
    Always-on background worker that continuously listens to an incoming email queue
    and processes emails through the LangChain EmailPipeline.
    """

    def __init__(self, pipeline: EmailPipeline, on_analyzed_callback: Optional[Callable[[EmailAnalysisOutput], None]] = None):
        self.pipeline = pipeline
        self.queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.on_analyzed_callback = on_analyzed_callback
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Starts the background listening loop."""
        if self.is_running:
            return
        self.is_running = True
        self._worker_task = asyncio.create_task(self._process_loop())
        logger.info("Always-On Email Watcher started.")

    async def stop(self):
        """Stops the background worker."""
        self.is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Always-On Email Watcher stopped.")

    async def push_email(self, email_dict: dict):
        """Enqueues an incoming email for immediate asynchronous analysis."""
        await self.queue.put(email_dict)

    async def _process_loop(self):
        while self.is_running:
            try:
                email_data = await self.queue.get()
                result = self.pipeline.analyze(
                    content=email_data.get("content", ""),
                    sender=email_data.get("sender", "Unknown"),
                    subject=email_data.get("subject", "")
                )
                if self.on_analyzed_callback:
                    if asyncio.iscoroutinefunction(self.on_analyzed_callback):
                        await self.on_analyzed_callback(result)
                    else:
                        self.on_analyzed_callback(result)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing email in watcher loop: {e}", exc_info=True)
