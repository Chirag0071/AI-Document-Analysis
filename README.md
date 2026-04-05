
# 📄 AI-Powered Document Analysis API

> **HCL Hackathon — Track 2: AI-Powered Document Analysis & Extraction**

An intelligent, production-grade document processing system that accepts **PDF**, **DOCX**, and **image** files and runs a full AI + ML + NLP pipeline to automatically extract key information.

---

## 🔗 Submission Links

| Item | Link |
|------|------|
| GitHub Repo | https://github.com/Chirag0071/AI-Document-Analysis |
| Live API URL | https://ai-document-analysis.onrender.com |
| API Endpoint | https://ai-document-analysis.onrender.com/api/document-analyze |
| Swagger Docs | https://ai-document-analysis.onrender.com/docs |
| Health Check | https://ai-document-analysis.onrender.com/health |
| API Key | `sk_track2_987654321` |

---

## 📌 Description

This API accepts a single Base64-encoded document and returns a rich JSON response including:

| Field | Description |
|-------|-------------|
| `summary` | AI-generated 2-4 sentence document summary |
| `entities` | Names, dates, organizations, locations, amounts, emails, phones, URLs |
| `sentiment` | Positive / Neutral / Negative |
| `sentiment_scores` | Per-class probability scores |
| `document_type` | Classified document category (Invoice, Resume, Report, etc.) |
| `key_phrases` | Top-10 most relevant keyphrases |
| `document_stats` | Word count, readability scores, language detection |
| `processing_time_seconds` | Time taken to process the document |

---

## 🛠️ Tech Stack

### Backend
| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | Core language |
| Framework | FastAPI | REST API framework |
| Server | Uvicorn | ASGI server |

### AI / LLM
| Tool | Model | Purpose |
|------|-------|---------|
| Groq API | `llama-3.3-70b-versatile` | Primary: summary + entities + sentiment (one call) |
| FinBERT | `ProsusAI/finbert` | Transformer-based sentiment analysis |

### ML / NLP
| Tool | Purpose |
|------|---------|
| spaCy `en_core_web_lg` | Named Entity Recognition (PERSON, ORG, GPE, DATE, MONEY) |
| scikit-learn RandomForest | Document type classification (10 categories) |
| scikit-learn DecisionTree | Sentiment meta-classifier combining 4 model scores |
| scikit-learn RandomForest | NER post-filter (rejects false positives) |
| VADER | Rule-based sentiment analysis |
| TextBlob | Pattern-based sentiment analysis |
| YAKE | Unsupervised keyphrase extraction |
| TF-IDF | N-gram keyphrase scoring + TextRank extractive summarisation |

### Document Extraction
| Format | Tools |
|--------|-------|
| PDF | pdfplumber (primary) → PyMuPDF (fallback) → Tesseract OCR (last resort) |
| DOCX | python-docx (paragraphs + tables + headers + footers) |
| Image | OpenCV 7-step preprocessing → Tesseract OCR (PSM 6 + PSM 3 + PSM 11) |

---

## 🤖 AI Tools Used

> **Mandatory AI Tool Policy disclosure — all AI assistance listed below**

| Tool | How Used |
|------|---------|
| **Groq (Llama 3.3 70B)** | Primary AI engine — generates summary, extracts entities, classifies sentiment in one API call |
| **FinBERT** (`ProsusAI/finbert`) | Deep learning transformer sentiment analysis (finance-aware) |
| **spaCy** `en_core_web_lg` | ML-based Named Entity Recognition |
| **VADER** | Rule-based sentiment model (model 1 of 4 in ensemble) |
| **TextBlob** | Pattern-based NLP sentiment (model 2 of 4) |
| **YAKE** | Unsupervised statistical keyphrase extraction |
| **scikit-learn** | Random Forest (document classifier + NER filter) + Decision Tree (sentiment meta-classifier) |

---

## 🏗️ Architecture Overview

```
POST /api/document-analyze
         │
    [Auth Check]
    x-api-key header → SHA-256 constant-time comparison → 401 if invalid
         │
    [Text Extraction]
    ┌────┴─────────────────────────────┐
    │ PDF  → pdfplumber                │
    │       → PyMuPDF (fallback)       │
    │       → Tesseract OCR (fallback) │
    │ DOCX → python-docx               │
    │        paragraphs + tables +     │
    │        headers + footers         │
    │ IMG  → OpenCV preprocessing      │
    │        (CLAHE + blur + thresh +  │
    │         morph + deskew +         │
    │         dark-sidebar detection)  │
    │       → Tesseract PSM 6/3/11     │
    └──────────────────────────────────┘
         │
    [Text Preprocessing]
    Unicode fix → OCR noise removal →
    Hyphenation fix → Whitespace norm
         │
    ┌────┴────────────────────────────────────────────────┐
    │                 AI + ML Pipeline                    │
    │                                                     │
    │  Groq LLM (primary)                                 │
    │  ┌───────────────────────────────────────────────┐  │
    │  │ ONE API call → summary + entities + sentiment │  │
    │  └───────────────────────────────────────────────┘  │
    │                       +                             │
    │  spaCy NER + Regex (always runs, merged with Groq)  │
    │  RF post-filter (rejects false positives)           │
    │                       +                             │
    │  Sentiment ensemble (VADER+TextBlob+FinBERT+DT)     │
    │  Decision Tree meta-classifier (combines 4 scores)  │
    │                       +                             │
    │  Random Forest document classifier (10 types)       │
    │  YAKE + TF-IDF keyphrase extractor                  │
    │  Readability stats (Flesch, FK, Gunning Fog)        │
    └─────────────────────────────────────────────────────┘
         │
    [JSON Response]
```

---

## 📁 Project Structure

```
AI-Document-Analysis/
│
├── main.py                  ← FastAPI app, auth, routing
├── .env.example             ← Template for environment variables
├── requirements.txt         ← All Python dependencies
├── test.py                  ← End-to-end test runner
├── README.md                ← This file
│
├── pipeline/
│   ├── __init__.py
│   └── processor.py         ← Master orchestrator (runs all 8 stages)
│
├── utils/
│   ├── __init__.py
│   ├── extractor.py         ← PDF/DOCX/Image text extraction
│   ├── preprocessor.py      ← Text cleaning and normalization
│   ├── summarizer.py        ← Groq LLM full analysis + extractive fallback
│   ├── entity.py            ← spaCy + regex + RF post-filter NER
│   ├── sentiment.py         ← 4-model ensemble + Decision Tree meta-classifier
│   └── stats.py             ← Readability metrics + document statistics
│
└── ml/
    ├── __init__.py
    ├── classifier.py        ← Random Forest document type classifier
    └── keyphrase.py         ← YAKE + TF-IDF hybrid keyphrase extractor
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Chirag0071/AI-Document-Analysis.git
cd AI-Document-Analysis
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_lg
```

### 5. Download TextBlob Corpora
```bash
python -m textblob.download_corpora
```

### 6. Install Tesseract OCR

**Windows:**
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Install to `C:\Program Files\Tesseract-OCR\`

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 7. Set Environment Variables
```bash
cp .env.example .env
```

Edit `.env`:
```
API_KEY=sk_track2_987654321
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com/

### 8. Run the Server
```bash
uvicorn main:app --reload
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/api/document-analyze` | Analyze a document |

### POST /api/document-analyze

**Headers**
```
Content-Type: application/json
x-api-key: sk_track2_987654321
```

**Request Body**
```json
{
  "fileName": "invoice.pdf",
  "fileType": "pdf",
  "fileBase64": "<base64 encoded file content>"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fileName` | string | Yes | Original file name |
| `fileType` | string | Yes | `pdf`, `docx`, or `image` |
| `fileBase64` | string | Yes | Base64-encoded file content |

### cURL Example
```bash
curl -X POST https://ai-document-analysis.onrender.com/api/document-analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_track2_987654321" \
  -d '{
    "fileName": "sample.pdf",
    "fileType": "pdf",
    "fileBase64": "JVBERi0xLjQ..."
  }'
```

### Success Response (200)
```json
{
  "status": "success",
  "fileName": "sample1-Technology_Industry_Analysis.pdf",
  "summary": "This document is an industry analysis report on the expansion of artificial intelligence innovation, highlighting key players such as Google, Microsoft, and NVIDIA across sectors including healthcare, finance, manufacturing, and education.",
  "entities": {
    "names": [],
    "dates": ["June 2020", "March 2017"],
    "organizations": ["Google", "Microsoft", "NVIDIA"],
    "locations": ["New York", "Brooklyn"],
    "amounts": ["₹10,000", "$5,000"],
    "percentages": ["30%", "25%"],
    "emails": [],
    "phones": [],
    "urls": []
  },
  "sentiment": "Positive",
  "sentiment_scores": {
    "Positive": 0.80,
    "Neutral": 0.15,
    "Negative": 0.05
  },
  "document_type": "Report / Article",
  "key_phrases": [
    "artificial intelligence",
    "machine learning",
    "technology companies"
  ],
  "document_stats": {
    "word_count": 412,
    "sentence_count": 18,
    "paragraph_count": 8,
    "character_count": 2841,
    "avg_sentence_length": 22.9,
    "avg_word_length": 5.2,
    "lexical_diversity": 0.612,
    "reading_time_minutes": 2.1,
    "language": "en",
    "readability": {
      "flesch_reading_ease": 42.3,
      "flesch_kincaid_grade": 13.1,
      "gunning_fog_index": 16.4,
      "interpretation": "Difficult"
    }
  },
  "processing_time_seconds": 3.24
}
```

### Error Responses

| HTTP Code | Reason |
|-----------|--------|
| 401 | Missing or invalid `x-api-key` header |
| 400 | Invalid base64 encoding or unsupported fileType |
| 413 | File exceeds 50 MB limit |
| 422 | No text could be extracted from document |
| 500 | Internal processing error |

---

## 🧩 Approach & Data Extraction Strategy

### 1. Text Extraction

**PDF — 3-layer fallback:**
1. `pdfplumber` — layout-aware extraction, preserves text positions
2. `PyMuPDF (fitz)` — faster, handles more PDF variants
3. Tesseract OCR — rasterizes each page at 2x zoom for scanned PDFs

**DOCX — Full document traversal:**
- All paragraphs with heading detection
- All table cells (rows joined with ` | `)
- Header and footer text (often contains company names, dates)

**Image — 7-step OpenCV pipeline:**
1. Decode image → BGR
2. Upscale images smaller than 1500px
3. Dark sidebar detection (for styled resumes)
4. Grayscale + CLAHE contrast enhancement
5. Gaussian blur (noise reduction)
6. Adaptive Gaussian thresholding (binarization)
7. Morphological close (reconnects broken characters)
8. Deskewing (fixes rotation in scanned documents)
9. Tesseract PSM 6, 3, 11 — picks longest result

### 2. Text Preprocessing

- Unicode normalization (NFC) + artifact fixes
- OCR garbage line removal (lines with <40% alphanumeric ratio)
- End-of-line hyphenation repair
- Structural marker removal
- Whitespace normalization

### 3. AI Summarisation

**Primary — Groq Llama 3.3 70B:**
- Single API call returns summary + entities + sentiment as JSON
- Temperature 0.1 for consistent factual output
- JSON validation with fallback if response is malformed

**Fallback — TF-IDF TextRank:**
- Splits text into sentences
- Computes TF-IDF score per sentence
- Selects top 3 sentences, preserving original order

### 4. Named Entity Recognition (3 layers)

**Layer 1 — Groq LLM:** Context-aware entity extraction via structured JSON prompt

**Layer 2 — spaCy `en_core_web_lg`:** Extracts PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT with:
- Stopword filter
- OCR noise filter (removes verbs like "transformed", "boosted")
- Fake org filter (removes sentence fragments)
- Random Forest post-filter (10 heuristic features)
- Validators: `_valid_name()`, `_valid_org()`, `_valid_date()`, `_valid_amount()`

**Layer 3 — 25+ Regex patterns:** Indian currency (₹, INR, Rs.), dates, phones, emails, URLs, percentages

All layers merged and deduplicated (case-insensitive, substring removal).

### 5. Sentiment Analysis (4-model ensemble)

| Model | Type |
|-------|------|
| VADER | Rule-based |
| TextBlob | Pattern-based |
| FinBERT/DistilBERT | Transformer DL |
| Decision Tree | Meta-classifier |

Feature vector (10 features) fed into Decision Tree meta-classifier. Strong agreement override: if VADER and TextBlob both agree, label is forced. Groq label takes precedence when available.

### 6. Document Classification

- Random Forest: 300 trees, balanced class weights
- TF-IDF: 800 features, bigrams, sublinear TF
- 10 document types, confidence threshold 0.30
- Trained on 250 synthetic documents

### 7. Keyphrase Extraction

```
final_score = 0.5 × YAKE + 0.4 × TF-IDF + 0.1 × position_bonus
```

YAKE (unsupervised, language-agnostic) + TF-IDF n-grams combined by hybrid reranker.

### 8. Readability Statistics

| Metric | Formula |
|--------|---------|
| Flesch Reading Ease | `206.835 - 1.015×ASL - 84.6×ASW` |
| Flesch-Kincaid Grade | `0.39×ASL + 11.8×ASW - 15.59` |
| Gunning Fog Index | `0.4×(ASL + pct_complex_words)` |
| Lexical Diversity | `unique_words / total_words` |

---

## 🔐 Authentication

All requests must include:
```
x-api-key: sk_track2_987654321
```

Uses **constant-time SHA-256 comparison** to prevent timing attacks. Returns `401 Unauthorized` for invalid keys.

---

## 🚀 Deployment on Render

### Step 1 — Push to GitHub
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

### Step 2 — Create Render Web Service
1. Go to https://render.com → Sign up / Login with GitHub
2. Click `New` → `Web Service`
3. Connect your GitHub repo: `Chirag0071/AI-Document-Analysis`
4. Configure settings:

| Setting | Value |
|---------|-------|
| Name | `ai-document-analysis` |
| Region | Singapore (closest to India) |
| Branch | `main` |
| Runtime | `Python 3` |
| Build Command | `pip install -r requirements.txt && python -m spacy download en_core_web_lg && python -m textblob.download_corpora` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free |

### Step 3 — Add Environment Variables
In Render dashboard → `Environment` tab:
```
API_KEY = sk_track2_987654321
GROQ_API_KEY = your_groq_api_key_here
```

### Step 4 — Deploy
Click `Create Web Service` → Wait 5-10 minutes for build

Your live URL will be:
```
https://ai-document-analysis.onrender.com
```

### Step 5 — Verify Deployment
```bash
curl https://ai-document-analysis.onrender.com/health
# Expected: {"status": "ok"}
```

### ⚠️ Render Free Tier Notes
- Service sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds (cold start)
- 512 MB RAM — torch/transformers may be heavy; they will fall back gracefully if memory is exceeded
- Add `PYTHONUNBUFFERED=1` as environment variable for better logs

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| Scanned PDFs | OCR accuracy depends on scan quality |
| Handwritten text | Tesseract not trained for handwriting |
| Non-English documents | Models are English-optimised |
| Large files | Truncated to 5,000 chars for Groq, 100,000 for spaCy |
| Cold start on Render | First request after inactivity takes ~30s |
| Processing time | Large PDFs with Groq: ~75s; DOCX/Image: 3-13s |

---

## 🧪 Testing

```bash
# Terminal 1 — start server
uvicorn main:app --reload

# Terminal 2 — run all 3 sample file tests
python test.py
```

---

## 📦 requirements.txt

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
python-dotenv==1.0.1
pdfplumber==0.11.0
PyMuPDF==1.24.5
python-docx==1.1.2
pytesseract==0.3.10
Pillow==10.3.0
opencv-python-headless==4.10.0.82
spacy==3.7.5
scikit-learn==1.5.0
numpy==1.26.4
groq
transformers==4.41.2
torch==2.3.0
vaderSentiment==3.3.2
textblob==0.18.0
yake==0.4.8
langdetect==1.0.9
pydantic==2.7.1
```

---

## 📜 License

Built for HCL Hackathon — Track 2. All rights reserved.

**Developer:** Chirag | GitHub: https://github.com/Chirag0071
