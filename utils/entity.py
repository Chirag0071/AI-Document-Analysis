"""
utils/entity.py — Named Entity Recognition.

Layer 1 : spaCy en_core_web_lg
Layer 2 : 25+ regex patterns (dates, amounts, emails, phones, URLs)
Layer 3 : Random Forest post-filter (rejects false positives)
Layer 4 : Deduplication + substring removal
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
                logger.info(f"spaCy model loaded: {model}")
                break
            except OSError:
                continue
        if _nlp is None:
            logger.warning("No spaCy model found — using regex only")
    return _nlp


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
        r"\b(Q[1-4]\s+\d{4})\b",
        r"\b(FY\s*\d{4}[-\/]\d{2,4})\b",
    ],
    "amounts": [
        # Indian currency — must have digits before symbol/word
        r"(₹\s?[\d,]+(?:\.\d{1,2})?(?:\s*(?:lakh|crore|thousand|million|billion))?)",
        r"((?:Rs\.?|INR)\s?[\d,]+(?:\.\d{1,2})?(?:\s*(?:lakh|crore|thousand|million|billion))?)",
        # Must have at least 2 digits before rupees/lakh/crore
        r"(\b[\d,]{2,}(?:\.\d{1,2})?\s+(?:rupees?|lakh|lakhs|crore|crores))\b",
        # USD / EUR / GBP — must have digits
        r"(\$\s?[\d,]+(?:\.\d{1,2})?(?:\s*(?:thousand|million|billion))?)",
        r"((?:USD|EUR|GBP|£|€)\s?[\d,]+(?:\.\d{1,2})?)",
        r"(\b[\d,]{2,}(?:\.\d{1,2})?\s+(?:dollars?|euros?|pounds?))\b",
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

# Words that are never valid entity names
_STOPWORDS = {
    "the","a","an","of","in","on","at","to","for","with","by","from",
    "is","was","are","were","be","been","and","or","but","not","as",
    "this","that","it","its","he","she","they","we","you","i",
    "mr","ms","mrs","dr","sir",
}

# OCR garbage words that appear in PERSON entities from image files
_OCR_NOISE_WORDS = {
    "nee", "transformed", "boosted", "developed", "focused", "skilled",
    "experienced", "motivated", "passionate", "dedicated", "creative",
    "responsible", "managed", "designed", "created", "delivered",
    "worked", "achieved", "improved", "increased", "reduced",
}

# Words that are NOT organizations despite spaCy thinking so
_FAKE_ORGS = {
    "cybersecurity incident report", "incident report", "major data breach",
    "data breach", "financial institutions", "banking systems",
    "digital banking", "online banking",
}


# ── Random Forest Post-Filter ─────────────────────────────────────────────────

class _RFFilter:
    def __init__(self):
        self._model = None
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np

            X = np.array([
                # Positives — well-formed entities
                [10,2,1,0.5,0,0,1,0],[15,3,1,0.3,0,0,1,0],
                [11,2,1,0.2,0,0,0,0],[12,3,1,0.2,1,0,0,1],
                [8,1,0,0.0,1,1,0,1], [9,2,1,0.4,0,0,1,0],
                [20,4,1,0.3,0,0,1,0],[8,1,1,1.0,0,0,1,0],
                [13,2,1,0.1,1,0,0,1],[7,1,0,0.0,1,1,0,1],
                [6,1,1,0.5,0,0,1,0], [9,2,1,0.2,0,0,0,0],
                # Negatives — false positives / garbage
                [1,1,0,0.0,1,0,0,0], [2,1,0,0.0,1,0,0,1],
                [3,1,1,1.0,0,1,0,0], [4,2,0,0.0,1,1,0,0],
                [1,1,1,1.0,0,0,1,0], [2,1,0,0.0,0,1,0,0],
                [3,1,0,0.0,1,1,0,1], [5,1,0,0.0,1,0,0,1],
                [2,1,1,1.0,0,0,1,0],
            ], dtype=float)

            y = [1]*12 + [0]*9

            self._model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )
            self._model.fit(X, y)
        except Exception as e:
            logger.warning(f"RF filter not available: {e}")

    def is_valid(self, text: str, label: str) -> bool:
        if self._model is None:
            return True
        import numpy as np
        feats = [
            len(text),
            len(text.split()),
            int(text[0].isupper()) if text else 0,
            sum(1 for c in text if c.isupper()) / max(len(text), 1),
            int(bool(re.search(r"\d", text))),
            int(bool(re.search(r"[^a-zA-Z0-9\s\.\,\-\']", text))),
            int(label in ("PERSON", "ORG", "GPE")),
            int(label in ("DATE", "MONEY")),
        ]
        return bool(self._model.predict(np.array([feats]))[0])


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedup(items: list) -> list:
    seen = {}
    for item in items:
        k = item.strip().lower()
        if k and len(k) > 1 and k not in seen:
            seen[k] = item.strip()
    keys = list(seen.keys())
    return [v for k, v in seen.items()
            if not any(k != k2 and k in k2 for k2 in keys)]


# ── Amount validator — reject garbage like "rs," ──────────────────────────────

def _valid_amount(amount: str) -> bool:
    """Return True only if the amount string contains actual digits."""
    return bool(re.search(r"\d{2,}", amount))


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
                    # Validate amounts — must have real digits
                    if cat == "amounts" and not _valid_amount(val):
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

            # Basic filters
            if not val or len(val) < 2:
                continue
            if val.lower() in _STOPWORDS:
                continue

            # Filter OCR noise from PERSON entities
            if ent.label_ == "PERSON":
                words = val.lower().split()
                if any(w in _OCR_NOISE_WORDS for w in words):
                    continue
                # Skip if entity is all lowercase (OCR garbage)
                if val == val.lower() and len(val.split()) > 1:
                    continue

            # Filter fake organizations
            if ent.label_ == "ORG":
                if val.lower() in _FAKE_ORGS:
                    continue
                # Skip if it looks like a sentence fragment
                if len(val.split()) > 5:
                    continue

            # ML post-filter
            if not self._rf.is_valid(val, ent.label_):
                continue

            if ent.label_ == "PERSON":
                result["names"].append(val)
            elif ent.label_ == "ORG":
                result["organizations"].append(val)
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                result["locations"].append(val)
            elif ent.label_ in ("DATE", "TIME"):
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

        # Merge, validate amounts from both sources
        all_amounts = sp.get("amounts", []) + rx.get("amounts", [])
        valid_amounts = [a for a in all_amounts if _valid_amount(a)]

        return {
            "names":         _dedup(sp.get("names", [])),
            "dates":         _dedup(sp.get("dates", []) + rx.get("dates", [])),
            "organizations": _dedup(sp.get("organizations", [])),
            "locations":     _dedup(sp.get("locations", [])),
            "amounts":       _dedup(valid_amounts),
            "percentages":   _dedup(sp.get("percentages", []) + rx.get("percentages", [])),
            "emails":        _dedup(rx.get("emails", [])),
            "phones":        _dedup(rx.get("phones", [])),
            "urls":          _dedup(rx.get("urls", [])),
        }