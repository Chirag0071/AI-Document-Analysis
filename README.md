# 📄 AI-Powered Document Analysis API

> **HCL GUVI BuildBridge Hackathon 2026 — Track 2: AI-Powered Document Analysis & Extraction**

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
| Groq API | `llama-3.3-70b-versatile` | Primary: summary + entities + sentiment in one call |
| FinBERT | `ProsusAI/finbert` | Transformer-based sentiment analysis |

### ML / NLP
| Tool | Purpose |
|------|---------|
| spaCy `en_core_web_lg` | Named Entity Recognition |
| scikit-learn RandomForest | Document type classification (10 categories) |
| scikit-learn DecisionTree | Sentiment meta-classifier |
| scikit-learn RandomForest | NER post-filter (rejects false positives) |
| VADER | Rule-based sentiment analysis |
| TextBlob | Pattern-based sentiment analysis |
| YAKE | Unsupervised keyphrase extraction |
| TF-IDF | N-gram keyphrase scoring + TextRank summarisation |

### Document Extraction
| Format | Tools |
|--------|-------|
| PDF | pdfplumber → PyMuPDF → Tesseract OCR |
| DOCX | python-docx (paragraphs + tables + headers + footers) |
| Image | OpenCV 7-step pipeline → Tesseract (PSM 6 + 3 + 11) |

---

## 🤖 AI Tools Used

> **Mandatory AI Tool Policy disclosure**

| Tool | How Used |
|------|---------|
| **Groq Llama 3.3 70B** | Primary AI — summary, entities, sentiment in one API call |
| **FinBERT** | Deep learning transformer sentiment analysis |
| **spaCy `en_core_web_lg`** | ML-based Named Entity Recognition |
| **VADER** | Rule-based sentiment (model 1 of 4) |
| **TextBlob** | Pattern-based sentiment (model 2 of 4) |
| **YAKE** | Unsupervised keyphrase extraction |
| **scikit-learn** | Random Forest + Decision Tree classifiers |

---

## 🏗️ Architecture

```
POST /api/document-analyze
         │
    [Auth] x-api-key → SHA-256 → 401 if invalid
         │
    [Extraction]
    PDF  → pdfplumber → PyMuPDF → OCR
    DOCX → python-docx
    IMG  → OpenCV pipeline → Tesseract
         │
    [Preprocessing]
    Unicode → OCR noise → Hyphenation → Whitespace
         │
    [AI + ML Pipeline]
    Groq LLM → summary + entities + sentiment (JSON)
    spaCy NER + Regex → merged with Groq entities
    VADER + TextBlob + FinBERT + DT → sentiment ensemble
    Random Forest → document type (10 categories)
    YAKE + TF-IDF → keyphrases
    Flesch + FK + Gunning Fog → readability
         │
    [JSON Response]
```

---

## 📁 Project Structure

```
AI-Document-Analysis/
│
├── main.py                  ← FastAPI app, auth, routing
├── .env                     ← API keys (never commit this)
├── .env.example             ← Template for environment variables
├── requirements.txt         ← All Python dependencies
├── README.md                ← This file
│
├── test.py                  ← Test with 3 embedded sample files
├── analyze_any.py           ← Analyze ANY file from your PC
│
├── pipeline/
│   ├── __init__.py
│   └── processor.py         ← Master orchestrator
│
├── utils/
│   ├── __init__.py
│   ├── extractor.py         ← PDF/DOCX/Image text extraction
│   ├── preprocessor.py      ← Text cleaning and normalization
│   ├── summarizer.py        ← Groq LLM + extractive fallback
│   ├── entity.py            ← spaCy + regex + RF NER
│   ├── sentiment.py         ← 4-model ensemble + DT
│   └── stats.py             ← Readability + statistics
│
└── ml/
    ├── __init__.py
    ├── classifier.py        ← Random Forest document classifier
    └── keyphrase.py         ← YAKE + TF-IDF keyphrase extractor
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Chirag0071/AI-Document-Analysis.git
cd AI-Document-Analysis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/macOS
```

### 3. Install All Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m textblob.download_corpora
```

### 4. Install Tesseract OCR
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki → Install to default path
- **Ubuntu:** `sudo apt-get install tesseract-ocr`
- **macOS:** `brew install tesseract`

### 5. Configure Environment
```bash
cp .env.example .env
```
Edit `.env`:
```
API_KEY=sk_track2_987654321
GROQ_API_KEY=your_groq_api_key_here
```
Get free Groq key at: https://console.groq.com/

### 6. Run Server
```bash
uvicorn main:app --reload
```

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/api/document-analyze` | Analyze a document |

### Request
```json
{
  "fileName": "invoice.pdf",
  "fileType": "pdf",
  "fileBase64": "<base64 string>"
}
```

### cURL
```bash
curl -X POST https://ai-document-analysis.onrender.com/api/document-analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_track2_987654321" \
  -d '{"fileName":"doc.pdf","fileType":"pdf","fileBase64":"JVBERi0..."}'
```

### Response (200)
```json
{
  "status": "success",
  "fileName": "doc.pdf",
  "summary": "This document is a technology industry analysis...",
  "entities": {
    "names": ["Ravi Kumar"],
    "dates": ["10 March 2026"],
    "organizations": ["ABC Pvt Ltd"],
    "locations": ["Mumbai"],
    "amounts": ["₹10,000"],
    "percentages": ["30%"],
    "emails": ["info@abc.com"],
    "phones": ["+91 9876543210"],
    "urls": []
  },
  "sentiment": "Positive",
  "sentiment_scores": {"Positive": 0.80, "Neutral": 0.15, "Negative": 0.05},
  "document_type": "Invoice / Receipt",
  "key_phrases": ["invoice", "payment due", "ABC Pvt Ltd"],
  "document_stats": {
    "word_count": 250,
    "sentence_count": 12,
    "reading_time_minutes": 1.3,
    "language": "en",
    "readability": {"flesch_reading_ease": 58.2, "interpretation": "Standard"}
  },
  "processing_time_seconds": 2.84
}
```

### Error Codes

| Code | Reason |
|------|--------|
| 401 | Invalid or missing x-api-key |
| 400 | Invalid base64 or unsupported fileType |
| 413 | File exceeds 50 MB |
| 422 | No text extractable |
| 500 | Internal processing error |

---

## 🧩 Approach & Strategy

### Text Extraction
- **PDF:** pdfplumber (layout-aware) → PyMuPDF (fallback) → Tesseract OCR (last resort for scanned PDFs)
- **DOCX:** python-docx reads all paragraphs, table cells, headers, footers
- **Image:** OpenCV pipeline — CLAHE contrast → Gaussian blur → adaptive threshold → morphological close → deskew → dark-sidebar detection → Tesseract with PSM 6, 3, 11

### Text Preprocessing
Unicode normalization → OCR garbage removal (lines <40% alphanumeric) → hyphenation repair → whitespace normalization

### AI Summarisation
1. Groq Llama 3.3 70B — ONE API call returns summary + entities + sentiment as structured JSON (temperature 0.1 for consistency)
2. TF-IDF TextRank extractive fallback (if Groq fails)

### Named Entity Recognition
- **Groq LLM:** Context-aware extraction via structured JSON prompt
- **spaCy en_core_web_lg:** PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT + Random Forest post-filter (rejects false positives) + validators (`_valid_name`, `_valid_org`, `_valid_date`, `_valid_amount`)
- **25+ Regex patterns:** Indian currency (₹, INR, Rs.), ISO dates, phones, emails, URLs, percentages
- All merged + case-insensitive deduplicated

### Sentiment Analysis
4-model ensemble with Decision Tree meta-classifier:
1. VADER (rule-based, chunk-averaged)
2. TextBlob (pattern-based polarity)
3. FinBERT/DistilBERT (transformer)
4. Decision Tree (10-feature meta-classifier)

Strong agreement override: if VADER and TextBlob both agree → force label. Groq label takes precedence when available.

### Document Classification
Random Forest (300 trees, balanced weights) + TF-IDF (800 features, bigrams) → 10 types with confidence threshold 0.30.

### Keyphrase Extraction
`score = 0.5×YAKE + 0.4×TF-IDF + 0.1×position_bonus`

### Readability
Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index, Lexical Diversity, Reading Time.

---

## 📂 Analyzing Any Document

You can analyze **any PDF, DOCX, or image** from your PC without changing any code:

### Interactive Menu
```bash
python analyze_any.py
```
Shows all supported files in folder — just type a number to analyze.

### Direct File Path
```bash
python analyze_any.py C:\Users\hp\Downloads\invoice.pdf
python analyze_any.py "C:\Users\hp\Desktop\my contract.docx"
python analyze_any.py C:\Users\hp\Pictures\receipt.jpg
```

### Supported Formats
| Extension | Type |
|-----------|------|
| `.pdf` | pdf |
| `.docx` `.doc` | docx |
| `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp` | image |

Results are printed in the terminal with colors. You can also save the full JSON result to a file.

---

## 🧪 Testing

```bash
# Terminal 1 — keep server running
uvicorn main:app --reload

# Terminal 2 — test with 3 sample files (embedded, no files needed)
python test.py

# Terminal 2 — analyze any file interactively
python analyze_any.py
```

---

## 🚀 Deployment on Render

### Step 1 — Push to GitHub
```bash
git add .
git commit -m "Final deployment"
git push origin main
```

### Step 2 — Create Render Web Service
1. Go to https://render.com → Login with GitHub
2. Click `New` → `Web Service`
3. Connect: `Chirag0071/AI-Document-Analysis`

### Step 3 — Configure

| Setting | Value |
|---------|-------|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python -m spacy download en_core_web_lg && python -m textblob.download_corpora` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free |

### Step 4 — Environment Variables
```
API_KEY = sk_track2_987654321
GROQ_API_KEY = your_groq_api_key
```

### Step 5 — Deploy
Click `Create Web Service` → Wait 5-10 minutes → URL is live!

> Render free tier sleeps after 15 min inactivity. First request after sleep takes ~30s.

---

## ⚠️ Known Limitations

| Issue | Detail |
|-------|--------|
| Scanned PDFs | OCR depends on scan quality |
| Handwritten text | Not supported by Tesseract |
| Non-English docs | English-optimised models |
| Large files | Truncated to 5,000 chars for Groq |
| Cold start | Render free: ~30s after inactivity |

---

## 🔐 Security

API key validated using constant-time SHA-256 comparison (prevents timing attacks). All files processed in memory — no files saved to disk.

---

## 📜 License

Built for **HCL GUVI BuildBridge Hackathon 2026** — Track 2.

**Developer:** Chirag | GitHub: https://github.com/Chirag0071