# Modular Small Language Model (SLM) Intelligence Platform

A modular architecture for fine-tuning a Small Language Model (SLM) on external accelerators (Google Colab, Kaggle, or GPU clusters) and deploying it with an always-on email intelligence pipeline, a map-reduce document summarizer, and a production-grade FastAPI backend containerized with Docker.

---

## 🏛️ Architecture & Separation of Concerns

Every component is decoupled into isolated modules:

```
slm/
├── llm/                          # Module 1: SLM Fine-Tuning & Local Engine
│   ├── config.py                 # Hyperparameters, LoRA ranks, model paths
│   ├── dataset.py                # Synthetic instruction datasets & chat templating
│   ├── finetune.py               # Standalone LoRA/PEFT fine-tuning script
│   └── model_loader.py           # Unified model loader & LangChain LLM wrapper
│
├── pipelines/                    # Module 2: LangChain Task Pipelines
│   ├── email_pipeline.py         # Email analysis (summary, action items, urgency, sentiment)
│   └── doc_summarizer.py         # Map-Reduce chunking summarizer for long documents
│
├── api/                          # Module 3: FastAPI Backend
│   ├── main.py                   # App lifecycle, model preloading, router registry
│   ├── schemas.py                # Pydantic request/response validation schemas
│   └── routes/
│       ├── email_routes.py       # POST /analyze-email & POST /enqueue-email
│       └── doc_routes.py         # POST /summarize-doc & POST /summarize-doc/upload
│
├── notebooks/                    # Module 4: Cloud / Accelerator Training
│   └── finetune_colab.ipynb      # 1-Click runnable notebook for Google Colab / Kaggle
│
├── tests/                        # Module 5: Automated Verification
│   ├── test_pipelines.py         # Pipeline tests
│   └── test_api.py               # FastAPI integration tests
│
├── Dockerfile                    # Containerization specification
├── docker-compose.yml            # Multi-service container orchestration
└── requirements.txt              # Project dependencies
```

---

## 🚀 Step 1: Model Fine-Tuning (Google Colab / Kaggle / GPU)

Rather than training from scratch or calling external cloud APIs, this system fine-tunes modern lightweight SLMs (e.g., `HuggingFaceTB/SmolLM-135M-Instruct` or `Qwen/Qwen2.5-0.5B-Instruct`) using **LoRA (Low-Rank Adaptation)** and **PEFT**.

### Option A: Using the Colab / Kaggle Notebook
1. Open [notebooks/finetune_colab.ipynb](file:///c:/Users/parme/OneDrive/Documents/slm/notebooks/finetune_colab.ipynb) in Google Colab or Kaggle.
2. Select **T4 GPU** runtime.
3. Run all cells: It will load the model, fine-tune it on the instruction dataset, and save the resulting LoRA adapter weights in `./fine_tuned_slm`.
4. Download the `fine_tuned_slm` directory and place it in `models/fine_tuned_slm` in this repository.

### Option B: Using the Command-Line Script
If you have an external GPU machine:
```bash
python -m llm.finetune \
    --model_id HuggingFaceTB/SmolLM-135M-Instruct \
    --output_dir ./models/fine_tuned_slm \
    --epochs 3 \
    --batch_size 4 \
    --lora_r 16 \
    --merge_adapter
```

---

## 📬 Step 2: Always-On Email Analysis Pipeline

Located at [pipelines/email_pipeline.py](file:///c:/Users/parme/OneDrive/Documents/slm/pipelines/email_pipeline.py):
- Extracts:
  - **Summary**: Concise 1–3 sentence synopsis.
  - **Action Items**: Explicit action items list.
  - **Urgency**: `Low` | `Medium` | `High` | `Critical`.
  - **Sentiment**: `Positive` | `Neutral` | `Negative`.
- **Always-on Watcher**: Includes `AlwaysOnEmailWatcher` background worker that continuously monitors an incoming queue and processes emails asynchronously.

---

## 📑 Step 3: On-Demand Map-Reduce Document Summarizer

Located at [pipelines/doc_summarizer.py](file:///c:/Users/parme/OneDrive/Documents/slm/pipelines/doc_summarizer.py):
- Triggers on-demand only when a document or file is submitted.
- Employs **Map-Reduce** chunking:
  1. **Split**: Splits long documents using `RecursiveCharacterTextSplitter`.
  2. **Map**: Summarizes each section independently.
  3. **Reduce**: Synthesizes intermediate summaries into a cohesive Executive Summary and key bulleted findings.
  4. Automatically bypasses map-reduce if the document fits in a single context window chunk.

---

## ⚡ Step 4: FastAPI Backend Serving

### Run Locally:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger UI will be live at: `http://localhost:8000/docs`

### Key Endpoints:

#### 1. Analyze Email (`POST /analyze-email`)
```bash
curl -X POST http://localhost:8000/analyze-email \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "secops@enterprise.com",
    "subject": "URGENT: Suspicious SSH activity on jump-host-01",
    "content": "Automated security alert: Multiple failed SSH attempts detected from unknown IP. Please isolate host and inspect auth.log before 15:00 UTC."
  }'
```
**Sample Response:**
```json
{
  "status": "success",
  "summary": "Multiple unauthorized SSH attempts detected on jump-host-01 requiring host isolation and log inspection by 15:00 UTC.",
  "action_items": [
    "Isolate jump-host-01 immediately.",
    "Inspect auth.log before 15:00 UTC."
  ],
  "urgency": "High",
  "sentiment": "Negative"
}
```

#### 2. Summarize Document (`POST /summarize-doc`)
```bash
curl -X POST http://localhost:8000/summarize-doc \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Technical Strategy",
    "text": "Your long document text goes here..."
  }'
```

#### 3. Upload Document File (`POST /summarize-doc/upload`)
```bash
curl -X POST http://localhost:8000/summarize-doc/upload \
  -F "file=@sample_report.txt"
```

---

## 🐳 Step 5: Docker Containerization

Deploy seamlessly using Docker:

```bash
# Build and run with Docker Compose
docker compose up --build -d

# Check service logs
docker compose logs -f

# Verify health status
curl http://localhost:8000/health
```
