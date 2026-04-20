"""
utils/entity.py — Named Entity Recognition.

Layer 1 : Groq LLM (via processor.py)
Layer 2 : spaCy en_core_web_lg
Layer 3 : 25+ regex patterns
Layer 4 : Random Forest post-filter
Layer 5 : Deduplication + garbage removal

"""

import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── spaCy ─────────────────────────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        for model in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
            try:
                import spacy
                _nlp = spacy.load(model)
                logger.info(f"spaCy loaded: {model}")
                break
            except OSError:
                continue
        if _nlp is None:
            logger.warning("No spaCy model — regex only")
    return _nlp


# ── Tech tool / framework / library blocklist ─────────────────────────────────
# spaCy frequently mis-tags these as PERSON or ORG. We block them explicitly.

_TECH_TOOLS = {
    # AI / ML frameworks & models
    "tensorflow","pytorch","keras","sklearn","scikit","scikit-learn",
    "xgboost","lightgbm","catboost","huggingface","transformers",
    "langchain","openai","anthropic","cohere","mistral","llama","llama2",
    "megatron","gpt","bert","roberta","albert","distilbert","t5","falcon",
    "stable diffusion","midjourney","dalle","whisper","clip","yolo",
    "opencv","pillow","pil","matplotlib","seaborn","plotly","bokeh",
    "mlflow","wandb","optuna","ray","dask","cudf","rapids","cuml",
    "onnx","triton","tensorrt","tflite","coreml","caffe","theano","mxnet",
    # Cloud / DevOps
    "docker","kubernetes","k8s","terraform","ansible","jenkins","github",
    "gitlab","bitbucket","circleci","travis","helm","istio","prometheus",
    "grafana","kibana","elasticsearch","logstash","fluentd",
    "aws","azure","gcp","heroku","vercel","netlify","cloudflare","nginx",
    "apache","gunicorn","uvicorn","celery","redis","rabbitmq","kafka",
    # Databases
    "mongodb","postgresql","mysql","sqlite","oracle","cassandra","dynamodb",
    "neo4j","pinecone","weaviate","qdrant","chroma","faiss","milvus",
    "snowflake","bigquery","redshift","hive","spark","hadoop","airflow",
    # Web frameworks
    "fastapi","flask","django","express","nextjs","nuxtjs","gatsby",
    "react","angular","vue","svelte","jquery","bootstrap","tailwind",
    "graphql","grpc","websocket","rest","restful","swagger","openapi",
    # Languages (often tagged as ORG by spaCy)
    "python","javascript","typescript","golang","rust","scala","kotlin",
    "swift","java","cplusplus","csharp","ruby","perl","matlab","julia",
    "bash","powershell","groovy","haskell","erlang","elixir","clojure",
    # Tools / Platforms
    "linux","ubuntu","debian","centos","fedora","macos","windows",
    "git","jira","confluence","slack","notion","trello","asana",
    "figma","sketch","invision","zeplin","adobe","photoshop","illustrator",
    "vscode","vim","emacs","jupyter","colab","kaggle","tableau","powerbi",
    "excel","powerpoint","word","sharepoint","salesforce","hubspot",
    # Specific tools flagged in testing
    "streamlit","gradio","dash","shiny","panel","voila",
    "pdfplumber","pymupdf","pytesseract","tesseract","spacy","nltk",
    "yake","textblob","vader","finbert","langdetect","textstat",
    "pydantic","sqlalchemy","alembic","httpx","aiohttp","requests",
    "numpy","pandas","scipy","statsmodels","networkx","igraph",
    "selenium","playwright","beautifulsoup","scrapy","puppeteer",
    # Acronyms spaCy tags as entities
    "api","sdk","ide","cli","gui","ui","ux","nlp","llm","rag","mlops",
    "devops","cicd","ci","cd","etl","elt","eda","poc","mvp","saas",
    "paas","iaas","vcs","orm","mvc","restapi","graphqlapi",
}

# ── Regex Patterns ────────────────────────────────────────────────────────────

_PATTERNS = {
    "dates": [
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)\s+\d{4})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b",
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    ],
    "amounts": [
        r"(₹\s?[\d,]+(?:\.\d{1,2})?(?:\s*(?:lakh|crore|thousand|million|billion))?)",
        r"((?:Rs\.?|INR)\s?[\d,]+(?:\.\d{1,2})?)",
        r"(\b[\d,]{3,}(?:\.\d{1,2})?\s+(?:rupees?|lakh|lakhs|crore|crores))\b",
        r"(\$\s?[\d,]+(?:\.\d{1,2})?(?:\s*(?:thousand|million|billion))?)",
        r"((?:USD|EUR|GBP|£|€)\s?[\d,]+(?:\.\d{1,2})?)",
    ],
    "percentages": [
        r"(\b\d{1,3}(?:\.\d{1,2})?\s*%)",
        r"(\b\d{1,3}(?:\.\d{1,2})?\s+percent)\b",
    ],
    "emails": [
        r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
    ],
    "phones": [
        r"(\+91[\s\-]?[6-9]\d{9})",
        r"(\b[6-9]\d{9}\b)",
        r"(\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4})",
    ],
    "urls": [
        r"(https?://[^\s<>\"]+)",
        r"(www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s]*)",
    ],
}

# Basic stopwords
_STOPWORDS = {
    "the","a","an","of","in","on","at","to","for","with","by","from",
    "is","was","are","were","be","been","and","or","but","not","as",
    "this","that","it","its","he","she","they","we","you","i",
    "mr","ms","mrs","dr","sir","none","null",
}

# OCR noise words that appear falsely in PERSON entities
_OCR_NOISE = {
    "transformed","boosted","developed","focused","skilled",
    "experienced","motivated","passionate","dedicated","creative",
    "responsible","managed","designed","created","delivered",
    "worked","achieved","improved","increased","reduced","enhanced",
    "implemented","coordinated","collaborated","mentoring","packaging",
    # Job title fragments spaCy falsely tags as names
    "engineer","developer","designer","analyst","scientist","manager",
    "director","officer","consultant","specialist","associate","intern",
    "senior","junior","lead","head","chief","president","vice",
    # Common OCR fragments
    "summary","objective","profile","skills","education","experience",
    "projects","achievements","contact","address","portfolio",
}

# Sentence fragments wrongly tagged as ORGs
_FAKE_ORGS = {
    "cybersecurity incident report","incident report","major data breach",
    "data breach","financial institutions","banking systems","digital banking",
    "online banking","cloud infrastructure","security researchers",
    "regulatory authorities","government agencies","technology experts",
    "financial analysts","cybersecurity experts","cybersecurity analysts",
    "machine learning","deep learning","natural language processing",
    "computer vision","artificial intelligence","data science",
    "open source","open-source","software development","web development",
}


# ── Random Forest Post-Filter ─────────────────────────────────────────────────

class _RFFilter:
    def __init__(self):
        self._model = None
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np

            X = np.array([
                [10,2,1,0.5,0,0,1,0],[15,3,1,0.3,0,0,1,0],
                [11,2,1,0.2,0,0,0,0],[12,3,1,0.2,1,0,0,1],
                [8,1,0,0.0,1,1,0,1], [9,2,1,0.4,0,0,1,0],
                [20,4,1,0.3,0,0,1,0],[8,1,1,1.0,0,0,1,0],
                [13,2,1,0.1,1,0,0,1],[6,1,1,0.5,0,0,1,0],
                [1,1,0,0.0,1,0,0,0], [2,1,0,0.0,1,0,0,1],
                [3,1,1,1.0,0,1,0,0], [4,2,0,0.0,1,1,0,0],
                [1,1,1,1.0,0,0,1,0], [2,1,0,0.0,0,1,0,0],
                [3,1,0,0.0,1,1,0,1], [2,1,1,1.0,0,0,1,0],
            ], dtype=float)
            y = [1]*10 + [0]*8

            self._model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )
            self._model.fit(X, y)
        except Exception as e:
            logger.warning(f"RF filter unavailable: {e}")

    def is_valid(self, text: str, label: str) -> bool:
        if self._model is None:
            return True
        import numpy as np
        feats = [
            len(text), len(text.split()),
            int(text[0].isupper()) if text else 0,
            sum(1 for c in text if c.isupper()) / max(len(text), 1),
            int(bool(re.search(r"\d", text))),
            int(bool(re.search(r"[^a-zA-Z0-9\s\.\,\-\']", text))),
            int(label in ("PERSON", "ORG", "GPE")),
            int(label in ("DATE", "MONEY")),
        ]
        return bool(self._model.predict(np.array([feats]))[0])


# ── Validators ────────────────────────────────────────────────────────────────

def _valid_amount(s: str) -> bool:
    return bool(re.search(r"\d{2,}", s))

def _valid_name(s: str) -> bool:
    """Must look like a real person name — not a tech tool or OCR noise."""
    words = s.split()
    if len(words) < 1 or len(words) > 5:
        return False
    if not words[0][0].isupper():
        return False
    # Reject if any word is a known tech tool (case-insensitive)
    sl = s.lower()
    if sl in _TECH_TOOLS:
        return False
    for w in words:
        if w.lower() in _TECH_TOOLS:
            return False
    # Reject OCR noise words
    if any(w.lower() in _OCR_NOISE for w in words):
        return False
    # Must not be all caps (OCR artifact) — except common abbreviations like "AI"
    if s == s.upper() and len(s) > 4:
        return False
    # Must be mostly letters
    alnum = sum(c.isalpha() or c.isspace() for c in s)
    if alnum / max(len(s), 1) < 0.8:
        return False
    # Reject single-word entries that are generic English words
    if len(words) == 1 and len(s) < 4:
        return False
    return True

def _valid_org(s: str) -> bool:
    """Must look like a real organization name."""
    sl = s.lower().strip()
    if sl in _FAKE_ORGS:
        return False
    # Single-word tech tools are NOT orgs (e.g. "TensorFlow", "Docker")
    words = s.split()
    if len(words) == 1 and sl in _TECH_TOOLS:
        return False
    # Reject very long entries (likely sentences)
    if len(words) > 8:
        return False
    # Reject if all words are tech tools (e.g. "Python Django Flask")
    if len(words) >= 2 and all(w.lower() in _TECH_TOOLS for w in words):
        return False
    return True

def _valid_date(s: str) -> bool:
    return bool(re.search(r"\b\d{4}\b|\b\d{1,2}[\/\-\.]\d{1,2}", s))


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedup(items: list) -> list:
    seen = {}
    for item in items:
        k = str(item).strip().lower()
        if k and len(k) > 1 and k not in seen:
            seen[k] = str(item).strip()
    keys = list(seen.keys())
    return [v for k, v in seen.items()
            if not any(k != k2 and k in k2 for k2 in keys)]


# ── Main Class ────────────────────────────────────────────────────────────────

class EntityExtractor:

    def __init__(self):
        self._rf = _RFFilter()

    def _regex(self, text: str) -> dict:
        result = defaultdict(list)
        for cat, patterns in _PATTERNS.items():
            for pat in patterns:
                for m in re.findall(pat, text, re.IGNORECASE):
                    val = m.strip()
                    if not val:
                        continue
                    if cat == "amounts" and not _valid_amount(val):
                        continue
                    if cat == "dates" and not _valid_date(val):
                        continue
                    result[cat].append(val)
        return result

    def _spacy(self, text: str) -> dict:
        result = defaultdict(list)
        nlp = _get_nlp()
        if not nlp:
            return result

        doc = nlp(text[:100_000])
        for ent in doc.ents:
            val = ent.text.strip()
            if not val or len(val) < 2:
                continue
            if val.lower() in _STOPWORDS:
                continue
            # Block tech tools early — before RF filter
            if val.lower() in _TECH_TOOLS:
                continue
            if not self._rf.is_valid(val, ent.label_):
                continue

            if ent.label_ == "PERSON":
                if _valid_name(val):
                    result["names"].append(val)
            elif ent.label_ == "ORG":
                if _valid_org(val):
                    result["organizations"].append(val)
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                result["locations"].append(val)
            elif ent.label_ in ("DATE", "TIME"):
                if _valid_date(val):
                    result["dates"].append(val)
            elif ent.label_ == "MONEY":
                if _valid_amount(val):
                    result["amounts"].append(val)
            elif ent.label_ == "PERCENT":
                result["percentages"].append(val)

        return result

    def extract(self, text: str) -> dict:
        sp = self._spacy(text)
        rx = self._regex(text)

        return {
            "names":         _dedup(sp.get("names", [])),
            "dates":         _dedup(sp.get("dates", []) + rx.get("dates", [])),
            "organizations": _dedup(sp.get("organizations", [])),
            "locations":     _dedup(sp.get("locations", [])),
            "amounts":       _dedup(sp.get("amounts", []) + rx.get("amounts", [])),
            "percentages":   _dedup(sp.get("percentages", []) + rx.get("percentages", [])),
            "emails":        _dedup(rx.get("emails", [])),
            "phones":        _dedup(rx.get("phones", [])),
            "urls":          _dedup(rx.get("urls", [])),
        }