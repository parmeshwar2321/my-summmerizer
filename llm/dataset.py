"""
Dataset preparation and formatting for multi-task instruction fine-tuning.
Handles:
1. Email analysis (summary, action items, urgency, sentiment extraction).
2. Document summarization (map-reduce style chunk summarization).
"""

import json
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict


# Comprehensive synthetic seed dataset for instruction fine-tuning
SAMPLE_INSTRUCTION_DATA: List[Dict[str, str]] = [
    # --- Task 1: Email Analysis ---
    {
        "system_prompt": (
            "You are an expert executive email analyst. Analyze the provided email and output ONLY "
            "a valid JSON object with the following keys: 'summary' (concise string), 'action_items' "
            "(list of strings), 'urgency' ('Low', 'Medium', 'High', or 'Critical'), and 'sentiment' "
            "('Positive', 'Neutral', or 'Negative'). Do not include any extra text or commentary."
        ),
        "user_input": (
            "From: sarah.connor@cyberdyne.io\n"
            "Subject: CRITICAL: Production DB Latency Spike & Data Leak Alert\n\n"
            "Team, our US-East database cluster is experiencing severe latency spikes exceeding 4500ms. "
            "More urgently, monitoring flagged anomalous outgoing data transfers at 02:15 UTC. "
            "We must immediately: 1. Rotate all master database credentials. 2. Isolate the affected replica node. "
            "3. Notify the compliance and security officer by 10 AM EST. "
            "Please jump on the bridge immediately."
        ),
        "response": json.dumps(
            {
                "summary": "Severe US-East database latency spike and potential data leak detected at 02:15 UTC requiring emergency containment.",
                "action_items": [
                    "Rotate all master database credentials immediately.",
                    "Isolate the affected replica node.",
                    "Notify compliance and security officer by 10 AM EST.",
                    "Join the emergency incident bridge call immediately."
                ],
                "urgency": "Critical",
                "sentiment": "Negative"
            },
            indent=2
        )
    },
    {
        "system_prompt": (
            "You are an expert executive email analyst. Analyze the provided email and output ONLY "
            "a valid JSON object with the following keys: 'summary' (concise string), 'action_items' "
            "(list of strings), 'urgency' ('Low', 'Medium', 'High', or 'Critical'), and 'sentiment' "
            "('Positive', 'Neutral', or 'Negative'). Do not include any extra text or commentary."
        ),
        "user_input": (
            "From: alex.miller@techcorp.com\n"
            "Subject: Great job on Q3 release! Next steps for design sync\n\n"
            "Hey everyone! Huge congratulations on shipping the Q3 feature set yesterday. "
            "Client feedback has been overwhelmingly positive. "
            "Whenever you have a chance next week, please review the Figma mockups for the Q4 roadmap "
            "and submit any feedback into the shared tracker before our Wednesday design review."
        ),
        "response": json.dumps(
            {
                "summary": "Celebration of successful Q3 release with positive feedback, requesting review of Q4 Figma designs by next Wednesday.",
                "action_items": [
                    "Review Figma mockups for Q4 roadmap.",
                    "Submit feedback to shared tracker prior to Wednesday design review."
                ],
                "urgency": "Low",
                "sentiment": "Positive"
            },
            indent=2
        )
    },
    {
        "system_prompt": (
            "You are an expert executive email analyst. Analyze the provided email and output ONLY "
            "a valid JSON object with the following keys: 'summary' (concise string), 'action_items' "
            "(list of strings), 'urgency' ('Low', 'Medium', 'High', or 'Critical'), and 'sentiment' "
            "('Positive', 'Neutral', or 'Negative'). Do not include any extra text or commentary."
        ),
        "user_input": (
            "From: vendor-billing@cloudservices.com\n"
            "Subject: Invoice #48291 Due in 5 Days\n\n"
            "Dear Finance Team,\n"
            "This is a standard reminder that monthly invoice #48291 for $12,450.00 is due on October 18th. "
            "Please ensure ACH transfer details are confirmed and receipt is forwarded to billing@cloudservices.com."
        ),
        "response": json.dumps(
            {
                "summary": "Routine payment reminder for invoice #48291 ($12,450.00) due on October 18th.",
                "action_items": [
                    "Confirm ACH transfer details.",
                    "Forward payment receipt to billing@cloudservices.com by October 18th."
                ],
                "urgency": "Medium",
                "sentiment": "Neutral"
            },
            indent=2
        )
    },
    # --- Task 2: Document Summarization ---
    {
        "system_prompt": (
            "You are a professional research and business document summarizer. Summarize the following "
            "text chunk clearly, highlighting key takeaways, core findings, and contextual implications."
        ),
        "user_input": (
            "Deep learning models, particularly Transformers with multi-head self-attention mechanisms, "
            "have revolutionized NLP. However, deploying multi-billion parameter models at scale imposes "
            "significant memory, compute, and latency costs. Small Language Models (SLMs) with fewer than 1 billion "
            "parameters have recently emerged as an efficient alternative. When trained on high-quality synthetic "
            "and curated domain-specific tokens, SLMs can achieve comparable accuracy on targeted tasks such as "
            "entity extraction, summarization, and sentiment classification, while operating within strict edge "
            "latency budgets and reduced operational expenditure."
        ),
        "response": (
            "Transformers offer state-of-the-art NLP performance, but multi-billion parameter models face steep "
            "latency and compute bottlenecks. Small Language Models (SLMs under 1B parameters) trained on high-quality "
            "domain data achieve competitive performance on specialized tasks like summarization and extraction while "
            "significantly reducing operational costs and latency."
        )
    },
    {
        "system_prompt": (
            "You are an executive assistant. Synthesize the following section summaries into a single cohesive, "
            "comprehensive executive summary."
        ),
        "user_input": (
            "Section 1 Summary: Enterprise cloud infrastructure migration achieved 99.99% availability while cutting operational cost by 22%.\n"
            "Section 2 Summary: Automated CI/CD pipelines reduced average deployment duration from 45 minutes to 7 minutes with zero security regressions.\n"
            "Section 3 Summary: Remaining challenges include cross-region database replication latency and developer onboarding learning curve."
        ),
        "response": (
            "The infrastructure transformation successfully increased uptime to 99.99% and slashed operating costs by 22%, "
            "supported by an optimized CI/CD pipeline that cut deployment times to 7 minutes. Future focus areas must resolve "
            "cross-region database latency and streamline developer onboarding."
        )
    }
]


def format_chat_prompt(system_prompt: str, user_input: str, response: Optional[str] = None, tokenizer=None) -> str:
    """
    Formats system, user, and assistant turns into standard chat prompt structure.
    Uses tokenizer chat template if available, otherwise falls back to ChatML format.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(response is None)
            )
        except Exception:
            pass

    # Standard ChatML fallback
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    text += f"<|im_start|>user\n{user_input}<|im_end|>\n"
    if response is not None:
        text += f"<|im_start|>assistant\n{response}<|im_end|>"
    else:
        text += "<|im_start|>assistant\n"
    return text


def load_instruction_dataset(
    custom_data_path: Optional[str] = None,
    tokenizer=None,
    test_size: float = 0.1
) -> DatasetDict:
    """
    Loads instruction tuning data, formats it into chat templates, and returns a DatasetDict.
    """
    data = SAMPLE_INSTRUCTION_DATA.copy()
    if custom_data_path:
        with open(custom_data_path, "r", encoding="utf-8") as f:
            if custom_data_path.endswith(".jsonl"):
                custom_data = [json.loads(line) for line in f if line.strip()]
            else:
                custom_data = json.load(f)
            data.extend(custom_data)

    formatted_records = []
    for item in data:
        text = format_chat_prompt(
            system_prompt=item["system_prompt"],
            user_input=item["user_input"],
            response=item.get("response", ""),
            tokenizer=tokenizer
        )
        formatted_records.append({"text": text})

    raw_dataset = Dataset.from_list(formatted_records)
    if len(raw_dataset) > 1 and test_size > 0:
        split = raw_dataset.train_test_split(test_size=test_size, seed=42)
        return split
    return DatasetDict({"train": raw_dataset, "test": raw_dataset})
