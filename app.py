"""
DocSense AI — FastAPI Backend
HCL GUVI BuildBridge Hackathon 2026 — Track 2

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os, re, base64, time, io, logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Optional heavy deps (graceful fallback if not installed) ─────────────────
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    import docx as python_docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import fitz          # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import yake
    HAS_YAKE = True
except ImportError:
    HAS_YAKE = False

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

try:
    from langdetect import detect as lang_detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# ─── Config ───────────────────────────────────────────────────────────────────
API_KEY        = os.getenv("API_KEY",       "sk_track2_987654321")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",  "")
GROQ_MODEL     = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")
MAX_SUMMARY_CH = 6000
MAX_FILE_MB    = 150

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("docsense")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocSense AI",
    description="AI-Powered Document Analysis — HCL GUVI BuildBridge Hackathon 2026 Track 2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve sample PDFs for the web frontend
SAMPLES_DIR = Path(__file__).parent / "samples"
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

# ─── Auth ─────────────────────────────────────────────────────────────────────
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/samples"):
        return await call_next(request)
    if request.headers.get("x-api-key","") != API_KEY:
        return JSONResponse(status_code=401,
            content={"status":"error","detail":"Unauthorized — invalid or missing x-api-key"})
    return await call_next(request)

# ─── Request model ────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    fileName:    str
    fileType:    str   # pdf | docx | image
    fileBase64:  str


# ════════════════════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def _decode(data: str) -> bytes:
    if "," in data:
        data = data.split(",", 1)[1]
    # safe pad
    data += "==" * ((4 - len(data) % 4) % 4)
    return base64.b64decode(data)


def _extract_pdf(raw: bytes) -> str:
    if HAS_PYMUPDF:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as exc:
        raise ValueError(f"PDF extraction failed: {exc}") from exc


def _extract_docx(raw: bytes) -> str:
    if not HAS_DOCX:
        raise ValueError("python-docx not installed — run: pip install python-docx")
    doc = python_docx.Document(io.BytesIO(raw))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_image(raw: bytes) -> str:
    if not HAS_OCR:
        raise ValueError("pytesseract / Pillow not installed")
    img = Image.open(io.BytesIO(raw))
    return pytesseract.image_to_string(img)


def extract_text(raw: bytes, file_type: str) -> str:
    ft = file_type.lower()
    if ft == "pdf":
        return _extract_pdf(raw)
    if ft in ("docx","doc"):
        return _extract_docx(raw)
    if ft == "image":
        return _extract_image(raw)
    raise ValueError(f"Unsupported fileType: {file_type}")


# ════════════════════════════════════════════════════════════════════════════════
#  DOCUMENT TYPE CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════

DOC_RULES = [
    (["invoice","bill","amount due","payment due","total amount","remit"],            "Invoice"),
    (["resume","curriculum vitae"," cv ","work experience","seeking internship"],     "Resume / CV"),
    (["report","analysis","findings","recommendations","executive summary"],          "Report"),
    (["contract","agreement","parties agree","terms and conditions","whereas"],       "Contract"),
    (["abstract","methodology","references","doi","journal","hypothesis"],            "Research Paper"),
    (["dear ","sincerely","regards,","to whom it may concern"],                       "Letter"),
    (["policy","insurance","coverage","premium","beneficiary","policyholder"],        "Policy Document"),
    (["receipt","order id","transaction id","purchase confirmed"],                    "Receipt / Order"),
    (["patient","diagnosis","prescription","dosage","medical history"],               "Medical Document"),
    (["agenda","minutes","resolution","motion","meeting held"],                       "Meeting Minutes"),
]

def classify_doc(text: str, filename: str) -> str:
    tl = (text[:3000] + " " + filename).lower()
    for keywords, label in DOC_RULES:
        if any(k in tl for k in keywords):
            return label
    return "General Document"


# ════════════════════════════════════════════════════════════════════════════════
#  AI SUMMARISATION (Groq → fallback extractive)
# ════════════════════════════════════════════════════════════════════════════════

def _extractive_summary(text: str) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
    return " ".join(sents[:3]) or text[:400]


def summarise(text: str, doc_type: str) -> str:
    if HAS_GROQ and GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role":"system","content":(
                        "You are an expert document analyst. "
                        "Write a concise 2-4 sentence summary of the document below. "
                        "Capture the main purpose, key facts, and any conclusions. "
                        "Plain prose only — no bullets, no markdown."
                    )},
                    {"role":"user","content":f"Document type: {doc_type}\n\n{text[:MAX_SUMMARY_CH]}"},
                ],
                temperature=0.25,
                max_tokens=280,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            log.warning(f"Groq summarisation failed: {exc}")
    return _extractive_summary(text)


# ════════════════════════════════════════════════════════════════════════════════
#  ENTITY EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

RE_EMAIL   = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_URL     = re.compile(r"https?://[^\s<>\"'`]+|www\.[^\s<>\"'`]+")
RE_PHONE   = re.compile(r"(\+?\d[\d\s\-().]{6,13}\d)")
RE_AMOUNT  = re.compile(
    r"[$₹€£¥]\s?\d[\d,]*\.?\d*"
    r"|\d[\d,]*\s?(?:USD|INR|EUR|GBP|AUD|SGD|crore|lakh|million|billion)",
    re.I
)
RE_PCT     = re.compile(r"\d+\.?\d*\s?%")
RE_DATE    = re.compile(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|\d{4}[/\-]\d{2}[/\-]\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s\d{1,2},?\s\d{4}"
    r"|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s\d{4})\b",
    re.I
)

def _unique(lst, limit=10):
    seen, out = set(), []
    for x in lst:
        x = x.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
            if len(out) >= limit:
                break
    return out


def extract_entities(text: str) -> dict:
    ents = {
        "names":[], "organizations":[], "dates":[],
        "locations":[], "amounts":[], "percentages":[],
        "emails":[], "phones":[], "urls":[],
    }

    ents["emails"]      = _unique(RE_EMAIL.findall(text))
    ents["urls"]        = _unique(RE_URL.findall(text))
    ents["amounts"]     = _unique(RE_AMOUNT.findall(text))
    ents["percentages"] = _unique(RE_PCT.findall(text))
    ents["dates"]       = _unique(RE_DATE.findall(text))

    # Phones — keep only plausible ones
    raw_phones = [re.sub(r"\s+","",m.group()) for m in RE_PHONE.finditer(text)]
    ents["phones"] = _unique(
        [p for p in raw_phones if 7 <= len(re.sub(r"\D","",p)) <= 15], limit=5
    )

    if HAS_SPACY:
        doc_nlp = nlp(text[:10000])
        names, orgs, locs = [], [], []
        for ent in doc_nlp.ents:
            v = ent.text.strip()
            if not v or len(v) < 2:
                continue
            if ent.label_ == "PERSON":
                names.append(v)
            elif ent.label_ in ("ORG","PRODUCT","WORK_OF_ART"):
                orgs.append(v)
            elif ent.label_ in ("GPE","LOC","FAC"):
                locs.append(v)
        ents["names"]         = _unique(names)
        ents["organizations"] = _unique(orgs)
        ents["locations"]     = _unique(locs)
    else:
        # Heuristic name extraction: consecutive capitalised words
        caps = re.findall(r"(?<![.!?\n])([A-Z][a-z]+ (?:[A-Z][a-z]+ ?){1,2})", text)
        ents["names"] = _unique(caps, 8)

    return ents


# ════════════════════════════════════════════════════════════════════════════════
#  SENTIMENT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def analyse_sentiment(text: str):
    chunk = text[:6000]
    pos = neu = neg = 0.0

    if HAS_VADER:
        vs = vader.polarity_scores(chunk)
        pos, neu, neg = vs["pos"], vs["neu"], vs["neg"]

    if HAS_TEXTBLOB:
        pol = TextBlob(chunk).sentiment.polarity
        b_pos = max(pol, 0.0)
        b_neg = abs(min(pol, 0.0))
        b_neu = max(1.0 - b_pos - b_neg, 0.0)
        if HAS_VADER:
            pos = (pos + b_pos) / 2
            neg = (neg + b_neg) / 2
            neu = (neu + b_neu) / 2
        else:
            pos, neg, neu = b_pos, b_neg, b_neu

    if not (HAS_VADER or HAS_TEXTBLOB):
        neu = 1.0

    total = pos + neu + neg or 1.0
    scores = {
        "Positive": round(pos / total, 4),
        "Neutral":  round(neu / total, 4),
        "Negative": round(neg / total, 4),
    }
    label = max(scores, key=scores.__getitem__)
    return label, scores


# ════════════════════════════════════════════════════════════════════════════════
#  KEY PHRASES
# ════════════════════════════════════════════════════════════════════════════════

def extract_keyphrases(text: str, n: int = 15) -> list:
    if HAS_YAKE:
        try:
            extractor = yake.KeywordExtractor(lan="en", n=2, top=n, dedupLim=0.7)
            return [kw for kw, _ in extractor.extract_keywords(text[:8000])]
        except Exception:
            pass
    # Fallback: frequency of capitalised bigrams + unigrams
    tokens = re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text)
    freq: dict = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return sorted(freq, key=freq.__getitem__, reverse=True)[:n]


# ════════════════════════════════════════════════════════════════════════════════
#  DOCUMENT STATS
# ════════════════════════════════════════════════════════════════════════════════

def readability(text: str) -> dict:
    if not HAS_TEXTSTAT:
        return {"flesch_reading_ease":"-","flesch_kincaid_grade":"-",
                "gunning_fog_index":"-","interpretation":"-"}
    try:
        fre = round(textstat.flesch_reading_ease(text), 1)
        fkg = round(textstat.flesch_kincaid_grade(text), 1)
        fog = round(textstat.gunning_fog(text), 1)
        interp = ("Easy" if fre >= 70 else "Standard" if fre >= 50
                  else "Difficult" if fre >= 30 else "Very Difficult")
        return {"flesch_reading_ease":fre,"flesch_kincaid_grade":fkg,
                "gunning_fog_index":fog,"interpretation":interp}
    except Exception:
        return {"flesch_reading_ease":"-","flesch_kincaid_grade":"-",
                "gunning_fog_index":"-","interpretation":"-"}


def detect_language(text: str) -> str:
    if HAS_LANGDETECT:
        try:
            return lang_detect(text[:500])
        except Exception:
            pass
    return "en"


def lexical_diversity(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    return round(len(set(words)) / len(words), 4) if words else 0.0


# ════════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["system"])
async def health():
    """Check API status and available components."""
    return {
        "status": "ok",
        "components": {
            "groq":      HAS_GROQ and bool(GROQ_API_KEY),
            "spacy":     HAS_SPACY,
            "ocr":       HAS_OCR,
            "yake":      HAS_YAKE,
            "vader":     HAS_VADER,
            "textblob":  HAS_TEXTBLOB,
            "textstat":  HAS_TEXTSTAT,
            "pymupdf":   HAS_PYMUPDF,
            "docx":      HAS_DOCX,
        }
    }


@app.post("/api/document-analyze", tags=["analysis"])
async def analyze(req: AnalyzeRequest):
    """
    Analyze a document and extract structured intelligence.

    **Authentication**: Provide `x-api-key` header.

    **Body**:
    - `fileName`: original file name
    - `fileType`: `pdf` | `docx` | `image`
    - `fileBase64`: base64-encoded file content
    """
    t0 = time.time()
    log.info(f"[analyze] {req.fileName} ({req.fileType})")

    # Decode
    try:
        raw = _decode(req.fileBase64)
    except Exception as exc:
        return JSONResponse(status_code=400,
            content={"status":"error","detail":f"Invalid base64: {exc}"})

    # Size guard
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        return JSONResponse(status_code=413,
            content={"status":"error","detail":f"File exceeds {MAX_FILE_MB} MB limit"})

    # Extract text
    try:
        text = extract_text(raw, req.fileType).strip()
    except Exception as exc:
        return JSONResponse(status_code=422,
            content={"status":"error","detail":str(exc)})

    if not text:
        return JSONResponse(status_code=422,
            content={"status":"error","detail":"No text could be extracted from this document."})

    # Pipeline
    doc_type              = classify_doc(text, req.fileName)
    summary               = summarise(text, doc_type)
    entities              = extract_entities(text)
    sentiment, sent_scores = analyse_sentiment(text)
    key_phrases           = extract_keyphrases(text)
    lang                  = detect_language(text)
    lex                   = lexical_diversity(text)
    rd                    = readability(text)

    words = len(re.findall(r"\b\w+\b", text))
    sents = max(len(re.split(r"[.!?]+", text)), 1)
    elapsed = round(time.time() - t0, 2)

    log.info(f"[analyze] done in {elapsed}s — {words} words, type={doc_type}, sentiment={sentiment}")

    return {
        "status":          "success",
        "fileName":        req.fileName,
        "document_type":   doc_type,
        "summary":         summary,
        "entities":        entities,
        "sentiment":       sentiment,
        "sentiment_scores":sent_scores,
        "key_phrases":     key_phrases,
        "document_stats": {
            "word_count":           words,
            "sentence_count":       sents,
            "reading_time_minutes": round(words / 200, 1),
            "language":             lang,
            "lexical_diversity":    lex,
            "readability":          rd,
        },
        "processing_time_seconds": elapsed,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)