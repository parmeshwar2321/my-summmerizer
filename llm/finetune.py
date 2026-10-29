"""
Fine-tuning script for Small Language Models (SLMs) using LoRA / PEFT.
Designed to be executed on Google Colab, Kaggle, or external GPU accelerators.

Usage:
    python -m llm.finetune --model_id HuggingFaceTB/SmolLM-135M-Instruct --epochs 3 --batch_size 4
"""

import argparse
import logging
import os
import sys
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer

from llm.config import ModelConfig, TrainingConfig
from llm.dataset import load_instruction_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SLMFineTuner")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an SLM using LoRA/PEFT")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM-135M-Instruct",
                        help="Hugging Face model ID or path to base model")
    parser.add_argument("--custom_data", type=str, default=None,
                        help="Path to custom JSON or JSONL instruction dataset")
    parser.add_argument("--output_dir", type=str, default="./models/fine_tuned_slm",
                        help="Directory to save fine-tuned LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit NF4 quantization")
    parser.add_argument("--merge_adapter", action="store_true", help="Merge LoRA adapter into base weights after training")
    return parser.parse_args()


def run_training():
    args = parse_args()
    logger.info("Initializing Fine-Tuning Pipeline...")
    logger.info(f"Target Model: {args.model_id}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Active GPU: {torch.cuda.get_device_name(0)}")

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Quantization Config
    bnb_config = None
    if args.load_in_4bit and torch.cuda.is_available():
        logger.info("Enabling 4-bit NormalFloat (NF4) quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    # 3. Base Model
    logger.info(f"Loading base model '{args.model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
        trust_remote_code=True
    )
    model.config.use_cache = False

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # 5. Dataset Loading
    logger.info("Loading instruction dataset...")
    dataset = load_instruction_dataset(
        custom_data_path=args.custom_data,
        tokenizer=tokenizer,
        test_size=0.1
    )
    logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['test'])}")

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        save_total_limit=2
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args
    )

    # 8. Train
    logger.info("Starting SFT training loop...")
    trainer.train()

    # 9. Save Weights & Tokenizer
    logger.info(f"Saving fine-tuned adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 10. Optional: Merge LoRA weights into base model for standalone serving
    if args.merge_adapter:
        merged_dir = os.path.join(args.output_dir, "merged")
        logger.info(f"Merging LoRA weights into base model and saving to {merged_dir}...")
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model successfully saved to {merged_dir}")

    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    run_training()
