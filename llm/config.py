"""
Configuration classes for Small Language Model (SLM) fine-tuning and inference.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for base model loading and LoRA adaptation."""

    # Default to modern high-efficiency SLMs (SmolLM 135M or Qwen2.5 0.5B)
    model_id_or_path: str = os.getenv("MODEL_ID_OR_PATH", "HuggingFaceTB/SmolLM-135M-Instruct")
    lora_adapter_path: Optional[str] = os.getenv("LORA_ADAPTER_PATH", None)

    # Device selection: 'auto', 'cuda', or 'cpu'
    device: str = os.getenv("DEVICE", "auto")

    # Quantization settings (useful for Colab / Kaggle T4 GPU)
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Inference defaults
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    repetition_penalty: float = 1.1

    # Context window limit
    max_context_length: int = 2048


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for PEFT/LoRA fine-tuning."""

    output_dir: str = "./models/fine_tuned_slm"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    fp16: bool = True  # set to True for GPU acceleration
    bf16: bool = False
    max_seq_length: int = 1024

    # LoRA Specific parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
