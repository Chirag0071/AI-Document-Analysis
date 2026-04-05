# 📄 AI-Powered Document Analysis API

> **HCL Hackathon — Track 2: AI-Powered Document Analysis & Extraction**

An intelligent, production-grade document processing system that accepts **PDF**, **DOCX**, and **image** files and runs a full AI + ML + NLP pipeline to automatically extract key information.

---

## 🔗 Submission Links

| Item | Link |
|------|------|
| Live API URL | `https://your-app.railway.app` |
| API Endpoint | `https://your-app.railway.app/api/document-analyze` |
| GitHub Repo | `https://github.com/your-username/hcl-doc-analyzer` |
| API Key | `sk_track2_987654321` |
| Swagger Docs | `https://your-app.railway.app/docs` |

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

> **Mandatory AI Tool Policy disclosure**

| Tool | How Used |
|------|---------|
| **Groq (Llama 3.3 70B)** | Primary AI engine — generates summary, extracts entities, classifies sentiment in one API call |
| **FinBERT** | Deep learning transformer sentiment analysis (finance-aware) |
| **spaCy `en_core_web_lg`** | ML-based Named Entity Recognition |
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
    ┌────┴──────────────────────────────────────────────────┐
    │                  AI + ML Pipeline                     │
    │                                                       │
    │  Groq LLM (primary)                                   │
    │  ┌─────────────────────────────────────────────────┐  │
    │  │ ONE API call → summary + entities + sentiment   │  │
    │  └─────────────────────────────────────────────────┘  │
    │                      +                                │
    │  spaCy NER + Regex (always runs, merged with Groq)    │
    │  RF post-filter (rejects false positives)             │
    │                      +                                │
    │  Sentiment ensemble (VADER + TextBlob + FinBERT + DT) │
    │  Decision Tree meta-classifier (combines 4 scores)    │
    │                      +                                │
    │  Random Forest document classifier (10 types)         │
    │  YAKE + TF-IDF keyphrase extractor                    │
    │  Readability stats (Flesch, FK, Gunning Fog)          │
    └───────────────────────────────────────────────────────┘
         │
    [JSON Response]
```

---

## 📁 Project Structure

```
HCL/
│
├── main.py                        ← FastAPI app, auth, routing
├── .env                           ← API keys (not committed)
├── .env.example                   ← Template for environment variables
├── requirements.txt               ← All Python dependencies
├── test.py                        ← End-to-end test runner (3 sample files embedded)
├── README.md                      ← This file
│
├── pipeline/
│   ├── __init__.py
│   └── processor.py               ← Master orchestrator (runs all 8 stages)
│
├── utils/
│   ├── __init__.py
│   ├── extractor.py               ← PDF/DOCX/Image text extraction
│   ├── preprocessor.py            ← Text cleaning and normalization
│   ├── summarizer.py              ← Groq LLM full analysis + extractive fallback
│   ├── entity.py                  ← spaCy + regex + RF post-filter NER
│   ├── sentiment.py               ← 4-model ensemble + Decision Tree meta-classifier
│   └── stats.py                   ← Readability metrics + document statistics
│
└── ml/
    ├── __init__.py
    ├── classifier.py              ← Random Forest document type classifier
    └── keyphrase.py               ← YAKE + TF-IDF hybrid keyphrase extractor
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/hcl-doc-analyzer.git
cd hcl-doc-analyzer
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

### 6. Install Tesseract OCR (system dependency)

**Windows:**
- Download installer: https://github.com/UB-Mannheim/tesseract/wiki
- Install with default settings to `C:\Program Files\Tesseract-OCR\`

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
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
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
curl -X POST https://your-app.railway.app/api/document-analyze \
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
- All paragraphs with heading detection (`[HEADING]` prefix)
- All table cells (rows joined with ` | `)
- Header and footer text (often contains company names, dates)

**Image — 7-step OpenCV pipeline:**
1. Decode image → BGR
2. Upscale images smaller than 1500px (better OCR accuracy)
3. Dark sidebar detection (for styled resumes with dark left panels)
4. Grayscale conversion
5. CLAHE contrast enhancement (handles uneven lighting)
6. Gaussian blur (reduces high-frequency noise)
7. Adaptive Gaussian thresholding (binarization)
8. Morphological close (reconnects broken character strokes)
9. Deskewing (fixes rotation in scanned documents)
10. Tesseract with PSM 6, PSM 3, PSM 11 — picks longest result

### 2. Text Preprocessing

After extraction, text goes through:
- Unicode normalization (NFC) + artifact fixes (smart quotes, em-dashes, ligatures)
- OCR garbage line removal (lines with <40% alphanumeric ratio)
- End-of-line hyphenation repair (`infor-\nmation` → `information`)
- Structural marker removal (`[HEADING]`, `[HEADER]`, `[FOOTER]`)
- Whitespace normalization (collapses 3+ blank lines to 2)

### 3. AI Summarisation (Groq LLM → Extractive fallback)

**Primary — Groq Llama 3.3 70B:**
- Single API call returns summary + entities + sentiment as structured JSON
- Prompt instructs the model to identify document type, key participants, amounts, dates
- Temperature 0.1 for consistent, factual output
- JSON validation with fallback extraction if response is malformed

**Fallback — TF-IDF TextRank:**
- Splits text into sentences
- Computes TF-IDF score per sentence
- Selects top 3 sentences by score, preserving original order

### 4. Named Entity Recognition (3-layer)

**Layer 1 — Groq LLM:** Extracts entities as part of the full analysis JSON call. Most accurate for context-aware extraction.

**Layer 2 — spaCy `en_core_web_lg`:** Extracts PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT entities. Results pass through:
- Stopword filter (removes common words)
- OCR noise filter (removes action verbs like "transformed", "boosted")
- Fake org filter (removes sentence fragments tagged as organizations)
- Random Forest post-filter (10 heuristic features, trained to reject false positives)
- Validators: `_valid_name()`, `_valid_org()`, `_valid_date()`, `_valid_amount()`

**Layer 3 — 25+ Regex patterns:** Catches:
- Indian currency: ₹, INR, Rs., lakh, crore
- ISO and natural-language dates
- International phone numbers (Indian mobile, +91, international)
- Email addresses, URLs
- Percentages

Results from all layers are merged and deduplicated (case-insensitive, substring removal).

### 5. Sentiment Analysis (4-model ensemble)

Four models vote, combined by Decision Tree meta-classifier:

| Model | Type | Weight |
|-------|------|--------|
| VADER | Rule-based | Primary |
| TextBlob | Pattern-based | Secondary |
| FinBERT/DistilBERT | Transformer DL | Tertiary |
| Decision Tree | Meta-classifier | Final arbiter |

**Feature vector (10 features):**
```
[vader_compound, vader_pos, vader_neg, vader_neu,
 textblob_polarity, textblob_subjectivity,
 sent_avg, sent_prop_pos, sent_prop_neg,
 transformer_score]
```

**Strong agreement override:** If VADER and TextBlob both agree (both negative or both positive), the label is forced regardless of DT output — prevents edge case misfires.

**Groq label takes precedence** when available (more context-aware for formal documents).

### 6. Document Classification (Random Forest)

- **300 trees**, max depth 12, balanced class weights
- **TF-IDF vectorizer**: 800 features, bigrams, sublinear TF, English stop words
- **10 document types**: Invoice/Receipt, Resume/CV, Report/Article, Contract/Agreement, Incident Report, Financial Statement, News Article, Academic/Research, Legal Document, General/Other
- Trained on 250 synthetic keyword-anchored documents (5 seeds × 4 augmentations per class)
- Confidence threshold 0.30 — low-confidence → "General / Other"
- Keyword fallback when sklearn unavailable

### 7. Keyphrase Extraction (YAKE + TF-IDF Hybrid)

**YAKE (Yet Another Keyword Extractor):**
- Unsupervised, language-agnostic
- Statistical features: term frequency, co-occurrence, position
- Max 3-gram phrases, top 20 candidates

**TF-IDF N-gram scoring:**
- Unigrams, bigrams, trigrams from preprocessed sentences
- Inverse document frequency across sentences as pseudo-corpus

**Hybrid reranker:**
```
final_score = 0.5 × YAKE_score + 0.4 × TFIDF_score + 0.1 × position_bonus
```
Position bonus: phrases in first 10% of document score +0.2

### 8. Document Statistics & Readability

| Metric | Formula |
|--------|---------|
| Flesch Reading Ease | `206.835 - 1.015×avg_sent_len - 84.6×avg_syllables` |
| Flesch-Kincaid Grade | `0.39×avg_sent_len + 11.8×avg_syllables - 15.59` |
| Gunning Fog Index | `0.4×(avg_sent_len + pct_complex_words)` |
| Lexical Diversity | `unique_words / total_words` |
| Reading Time | `word_count / 200` (avg reading speed) |

---

## 🔐 Authentication

All requests must include:
```
x-api-key: sk_track2_987654321
```

Implementation uses **constant-time SHA-256 comparison** to prevent timing attacks:
```python
provided = hashlib.sha256(x_api_key.encode()).digest()
expected = hashlib.sha256(API_KEY.encode()).digest()
if provided != expected:
    raise HTTPException(status_code=401)
```

Requests without a valid key receive `401 Unauthorized` immediately.

---

## 🚀 Deployment

### Railway (recommended)
1. Push code to GitHub
2. Connect repo at `railway.app`
3. Add environment variables: `API_KEY`, `GROQ_API_KEY`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Generate domain → live URL

### Render / Fly.io / AWS Free Tier
Same process — any platform supporting Python + uvicorn works.

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| Scanned PDFs | OCR accuracy depends on scan quality; very low-res scans may produce garbage text |
| Handwritten text | Tesseract not trained for handwriting — accuracy very low |
| Non-English documents | spaCy and VADER are English-optimised; other languages will have lower accuracy |
| Large files | Text truncated to 5,000 chars for Groq, 6,000 for summariser, 100,000 for spaCy |
| Dark sidebar images | Special handling added but may still miss text in very dark regions |
| Processing time | PDF with Groq: ~75s for large docs (Groq rate limit); DOCX/Image: 3-13s |

---

## 📦 Installation — Full Command Reference

```bash
# All libraries in one command
pip install fastapi uvicorn[standard] python-multipart python-dotenv \
    pdfplumber PyMuPDF python-docx pytesseract Pillow \
    opencv-python-headless spacy scikit-learn numpy \
    groq transformers torch vaderSentiment textblob \
    yake langdetect pydantic

# spaCy model
python -m spacy download en_core_web_lg

# TextBlob corpora
python -m textblob.download_corpora
```

---

## 🧪 Testing

Run the built-in test script (3 sample files embedded as base64):
```bash
# Terminal 1 — start server
uvicorn main:app --reload

# Terminal 2 — run tests
python test.py
```

Expected output per file:
- `STATUS: success`
- `SUMMARY`: 2-4 sentence AI-generated paragraph
- `SENTIMENT`: Positive / Negative / Neutral
- `ORGANIZATIONS`, `NAMES`, `DATES`, `AMOUNTS` as applicable

---

## 📜 License

Built for HCL Hackathon purposes. All rights reserved.
