"""
utils/entity.py — Named Entity Recognition.

Layer 1 : Groq LLM (via processor.py)
Layer 2 : spaCy en_core_web_lg
Layer 3 : 25+ regex patterns
Layer 4 : Random Forest post-filter
Layer 5 : Deduplication + garbage removal

FIX LOG v3 (based on live resume OCR output):
  - _TECH_TOOLS: 200+ entry blocklist stops tech tools appearing as PERSON/ORG.
  - _LOCATION_BLOCKLIST: blocks social platforms & OCR garbage from locations
    (YouTube, Gahub, Langcham, Stream, Github, etc.)
  - _JOB_TITLE_WORDS: words that indicate a string is a job role, not an org
    (Tech Stack, Data Scientist, Senior, Mentor, Team, etc.)
  - _valid_name(): rejects OCR-garbled multi-word phrases — requires names to
    have ONLY title-case single words (no lowercase interior words, no numbers,
    no special chars like &, -, |). Max 3 words for a person name.
  - _valid_org(): rejects strings containing job-title words, date fragments,
    symbols like & and |, and strings that are clearly project names or descriptions.
  - _valid_location(): new function — rejects social platforms, tech tools,
    OCR words, and single words that are clearly not places.
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
    # OCR aliases — common Tesseract mangling of tech tool names
    "keres",       # keras
    "toreh",       # pytorch
    "pythan",      # python
    "pal","palm",  # PaLM model
    "ineuron",     # iNeuron (platform, not a person/place)
    "docblend",    # DocBlend Hub
    "fleabsty",    # OCR garbage
    "destébert","destebert",  # DistilBERT
    "carcie",      # CircleCI
    "moviepy","moviepython",  # MoviePy
    "gahub","gaahub",         # GitHub OCR variants
    "langcham",               # LangChain OCR variant
    "customtinker",           # CustomTkinter
    "nlpx","nlpxtransformer", # OCR garbage
    "vectorstores","vectorstore", # vector DB tool
    "evidently",   # Evidently AI
    "elmo","roberta","albert","t5model", # more model names
}

# ── Location blocklist ────────────────────────────────────────────────────────
# spaCy tags social platforms, tech tools, and OCR garbage as GPE/LOC.

_LOCATION_BLOCKLIST = {
    # Social / streaming platforms
    "youtube","github","gahub","gitlab","linkedin","twitter","facebook",
    "instagram","medium","kaggle","twitch","tiktok","discord","telegram",
    "stream","streams","langcham","langchain","huggingface",
    # Tech tools spaCy tags as locations
    "docker","kubernetes","aws","azure","gcp","heroku","vercel",
    "mongodb","postgresql","mysql","redis","kafka","spark","hadoop",
    "python","javascript","typescript","react","angular","vue",
    "tensorflow","pytorch","keras","bert","gpt","llm","nlp","rag",
    "fastapi","flask","django","express","nodejs","node",
    "opencv","pandas","numpy","scipy","sklearn","xgboost",
    "mlflow","wandb","airflow","celery","nginx","apache",
    # OCR garbage words
    "gahub","langcham","ineuron","docblend","fleabsty","customtinker",
    "vectorstores","nlpx","destébert","carcie","heelthcare",
    # Generic non-location words
    "stack","backend","frontend","fullstack","cloud","platform",
    "model","models","data","code","app","application","system",
    "web","mobile","api","service","pipeline","framework",
}

# ── Job-title word set for ORG validation ─────────────────────────────────────
# If an entity contains these words it's a job description, not an org name.

_JOB_TITLE_WORDS = {
    "senior","junior","lead","head","chief","principal","staff",
    "data scientist","data science","machine learning","deep learning",
    "software engineer","data analyst","product manager","project manager",
    "tech stack","technology stack","full stack","frontend","backend",
    "mentor","mentee","team lead","team member","intern","associate",
    "present","current","ongoing","oct'","jan'","feb'","mar'","apr'",
    "may'","jun'","jul'","aug'","sep'","nov'","dec'",
    "nup","hlp","chabot","chatbot","vlp","nlpx","poc",
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
    "nee","transformed","boosted","developed","focused","skilled",
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
    # Words that NEVER appear in a real human name
    "tech","stack","hub","core","model","layer","module","service",
    "system","platform","cloud","data","api","web","app","tool",
    "framework","library","network","pipeline","workflow","engine",
    "team","group","division","department","unit","center","lab",
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
    """Must look like a real human name — strict rules to eliminate OCR garbage."""
    s = s.strip()
    words = s.split()

    # Person names are 1–3 words max
    if len(words) < 1 or len(words) > 3:
        return False

    # Every word must start with a capital letter
    if not all(w[0].isupper() for w in words if w):
        return False

    # No lowercase interior words (eliminates "Qeneraton exclusively Lalored",
    # "Reect Visdum AI Application", "Video Summanzaton" etc.)
    for w in words:
        if len(w) > 1 and not w[0].isupper():
            return False

    # No special characters (eliminates "Data Scientist & Mentor", "NUP - LSTM")
    if re.search(r"[&|\-\+\(\)\[\]\/\\@#\$\*\d]", s):
        return False

    # All chars must be letters, spaces, or apostrophes (for O'Brien etc.)
    if not re.match(r"^[A-Za-z\s\'\.]+$", s):
        return False

    # Reject OCR-corruption suffix patterns (manzaton, aton, visdum, reect etc.)
    _ocr_suffix = re.compile(
        r"(aton|eton|izaton|manzaton|visdum|reect|qener|laored|"
        r"fleab|sty$|nzat|mmanz|ummar)$",
        re.IGNORECASE
    )
    for w in words:
        if _ocr_suffix.search(w):
            return False

    # Reject tech tools
    sl = s.lower()
    if sl in _TECH_TOOLS:
        return False
    for w in words:
        if w.lower() in _TECH_TOOLS:
            return False

    # Reject OCR noise / job title words
    if any(w.lower() in _OCR_NOISE for w in words):
        return False

    # Reject all-caps entries (OCR abbreviations) — allow 2-char initials
    if s == s.upper() and len(s) > 3:
        return False

    # Must be mostly letters
    alnum = sum(c.isalpha() or c.isspace() for c in s)
    if alnum / max(len(s), 1) < 0.85:
        return False

    # Each word must be at least 2 chars (eliminates single-letter noise)
    if any(len(w) < 2 for w in words):
        return False

    return True

def _valid_org(s: str) -> bool:
    """Must look like a real organization — not a job title, tech stack, or description."""
    sl = s.lower().strip()

    # Explicit fake org fragments
    if sl in _FAKE_ORGS:
        return False

    words = s.split()

    # Too long = sentence fragment
    if len(words) > 6:
        return False

    # Reject strings with date fragments like "Oct'22 - Present"
    if re.search(r"(oct|jan|feb|mar|apr|may|jun|jul|aug|sep|nov|dec)'?\s*\d{2}", sl):
        return False
    if re.search(r"\d{4}\s*[-–]\s*(present|current|\d{4})", sl):
        return False

    # Reject if it contains job-title indicator words
    for jt in _JOB_TITLE_WORDS:
        if jt in sl:
            return False

    # Reject strings that are clearly tech descriptions (contain & | between tech terms)
    if re.search(r"\b(lstm|bert|gpt|rag|nlp|llm|mlops|dvc|mlflow)\b", sl):
        return False

    # Single-word tech tools are NOT orgs
    if len(words) == 1 and sl in _TECH_TOOLS:
        return False

    # If all words are tech tools → not an org
    if len(words) >= 2 and all(w.lower() in _TECH_TOOLS for w in words):
        return False

    # Reject strings starting with common non-org words
    first = words[0].lower() if words else ""
    if first in {"tech","data","machine","deep","natural","computer","software",
                 "open","full","front","back","web","mobile","cloud","core",
                 "hlp","nup","vlp","nlpx","poc","deta"}:
        return False

    return True

def _valid_location(s: str) -> bool:
    """Must be an actual geographic place — not a platform, tool, or OCR word."""
    sl = s.lower().strip()

    # Explicit blocklist
    if sl in _LOCATION_BLOCKLIST:
        return False

    # Tech tools in location position
    if sl in _TECH_TOOLS:
        return False

    # Single short words that aren't real places (OCR noise)
    words = s.split()
    if len(words) == 1:
        # Allow known country/city single words but reject anything suspicious
        if len(s) < 3:
            return False
        # Reject words with numbers or special chars
        if re.search(r"[\d&|@#\$\*\+\(\)\[\]]", s):
            return False
        # Reject all-lowercase single words (OCR garbage)
        if s == s.lower() and s not in {"india","usa","uk","uae","us"}:
            return False

    # Reject if it contains obvious tech indicators
    if re.search(r"\b(api|sdk|db|ml|ai|nlp|llm|gpt|bert|url|http)\b", sl):
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
                if _valid_location(val):
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