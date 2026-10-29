"""
Model Loader and LangChain LLM Wrapper.
Provides a unified interface for loading local or Hugging Face SLM checkpoints
and wrapping them seamlessly into LangChain Runnables/LLMs.
"""

import json
import logging
import os
import sys
from typing import Any, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline,
)
from peft import PeftModel

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from llm.config import ModelConfig

logger = logging.getLogger("SLMModelLoader")


class LocalSLMLangChainWrapper(LLM):
    """
    Custom LangChain LLM wrapper around a local Hugging Face CausalLM pipeline.
    Ensures consistent generation parameters, prompt formatting, and error handling.
    """

    generator: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

    @property
    def _llm_type(self) -> str:
        return "slm_local"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.generator is None:
            raise ValueError("Model generator is not initialized.")

        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "do_sample": kwargs.get("temperature", self.temperature) > 0.0,
            "return_full_text": False,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        outputs = self.generator(prompt, **generation_kwargs)
        generated_text = outputs[0]["generated_text"]

        if stop:
            for s in stop:
                if s in generated_text:
                    generated_text = generated_text.split(s)[0]

        return generated_text.strip()


class MockLangChainWrapper(LLM):
    """
    Lightweight deterministic mock LLM used for isolated testing and offline development
    when model weights are not loaded.
    """

    @property
    def _llm_type(self) -> str:
        return "mock_slm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt_lower = prompt.lower()
        if "action_items" in prompt_lower or "email" in prompt_lower:
            return json.dumps({
                "summary": "Urgent update regarding production infrastructure latency and security audit.",
                "action_items": [
                    "Isolate database replica node.",
                    "Rotate master access credentials immediately.",
                    "Notify compliance lead."
                ],
                "urgency": "High",
                "sentiment": "Negative"
            })
        elif "map" in prompt_lower or "section" in prompt_lower:
            return "Key chunk finding: System architecture optimized for low-latency edge inference."
        else:
            return "Executive Summary: The document provides comprehensive guidance on efficient SLM deployment and pipeline orchestration."


class SLMModelLoader:
    """
    Factory and manager for loading tokenizers, models, and wrapping them into LangChain.
    """

    _instance: Optional["SLMModelLoader"] = None
    _llm: Optional[LLM] = None

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self.pipeline: Optional[TextGenerationPipeline] = None

    @classmethod
    def get_instance(cls, config: Optional[ModelConfig] = None) -> "SLMModelLoader":
        """Singleton access to avoid reloading heavy model weights multiple times."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def load(self) -> LLM:
        """Loads model into memory and returns LangChain LLM wrapper."""
        if self._llm is not None:
            return self._llm

        # Allow mock mode via environment variable for fast unit tests or CI
        if os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes"):
            logger.info("MOCK_LLM enabled: Using deterministic MockLangChainWrapper.")
            self._llm = MockLangChainWrapper()
            return self._llm

        logger.info(f"Loading tokenizer and model from '{self.config.model_id_or_path}'...")
        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if device_str == "cuda" else torch.float32

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id_or_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device_str == "cuda" else None,
            trust_remote_code=True
        )
        if device_str == "cpu":
            self.model.to("cpu")

        # 3. Load optional LoRA adapter
        if self.config.lora_adapter_path and os.path.exists(self.config.lora_adapter_path):
            logger.info(f"Applying LoRA adapter from {self.config.lora_adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.config.lora_adapter_path)

        self.model.eval()

        # 4. Build HuggingFace Pipeline
        pipe_device = 0 if device_str == "cuda" else -1
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipe_device if device_str != "cuda" else None,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )

        # 5. Wrap in LangChain LLM
        self._llm = LocalSLMLangChainWrapper(
            generator=self.pipeline,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        logger.info("Model loaded and wrapped successfully into LangChain interface.")
        return self._llm

    def get_llm(self) -> LLM:
        """Returns loaded LangChain LLM, loading it if not already initialized."""
        if self._llm is None:
            return self.load()
        return self._llm
