"""
ml/keyphrase.py — Key-phrase extraction.

YAKE (unsupervised statistical) + TF-IDF n-gram hybrid reranker.
Falls back to frequency-based extraction if YAKE unavailable.

FIX LOG:
  - Added KP_STOP: a set of stop phrases that should never be returned
    as keyphrases (avoids "using", "video", "linkedin data scientist",
    "data scientist", etc. polluting the skills display).
  - Added _clean_phrase(): strips leading/trailing stopwords from
    multi-word keyphrases before returning.
  - Minimum phrase length raised from 4 → 5 chars to reduce noise.
  - Added max word count guard (≤ 5 words) to avoid sentence fragments.
"""

import re
import math
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the","a","an","of","in","on","at","to","for","with","by","from",
    "is","was","are","were","be","been","and","or","but","not","as",
    "this","that","these","those","it","its","he","she","they","we",
    "have","has","had","do","does","did","will","would","could","should",
    "which","who","when","where","how","what","also","more","most",
    "other","into","than","then","their","there","here","very","just",
}

# Keyphrases that look real but are noise / should never appear in skills
KP_STOP = {
    "video","senior","junior","lead","manager","engineer","scientist",
    "analyst","developer","designer","intern","candidate","professional",
    "specialist","experience","skills","education","projects","work",
    "role","team","years","data","kaggle","streamlit","youtube",
    "github","langchan","information","please","contact",
    "looking","seeking","seeking","motivated","passionate","dedicated",
    "creative","responsible","overview","objective","summary","profile",
    "reference","available","request","currently","previous","following",
    "including","related","various","different","multiple","several",
    # Job title fragments
    "data scientist","machine learning engineer","software engineer",
    "data analyst","product manager","project manager","full stack",
    "frontend","backend","frontend developer","backend developer",
    "data scientist",
}


def _clean_phrase(phrase: str) -> str:
    """Strip leading/trailing stopwords from a keyphrase."""
    words = phrase.split()
    while words and words[0].lower() in _STOPWORDS:
        words = words[1:]
    while words and words[-1].lower() in _STOPWORDS:
        words = words[:-1]
    return " ".join(words)


def _yake(text: str) -> list:
    try:
        import yake
        ex = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7,
                                   dedupFunc="seqm", windowsSize=1, top=25)
        kws = ex.extract_keywords(text[:8000])
        return [(kw, 1 - score) for kw, score in kws]
    except Exception:
        return []


def _tfidf_ngrams(text: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    all_tok = [[t for t in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", s.lower())
                if t not in _STOPWORDS]
               for s in sentences]
    N = len(sentences) or 1

    df = defaultdict(int)
    candidates = []
    for toks in all_tok:
        seen = set()
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i+n])
                candidates.append(g)
                if g not in seen:
                    df[g] += 1
                    seen.add(g)

    tf = Counter(candidates)
    total = sum(tf.values()) or 1
    scores = {p: (c/total) * math.log((N+1)/(df[p]+1))
              for p, c in tf.items() if len(p) >= 5}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:25]


def _normalize(items):
    if not items:
        return []
    mn, mx = min(s for _, s in items), max(s for _, s in items)
    rng = mx - mn or 1
    return [(p, (s-mn)/rng) for p, s in items]


def _is_valid_keyphrase(phrase: str) -> bool:
    """Return True if this phrase is worth keeping."""
    cleaned = _clean_phrase(phrase)
    if not cleaned or len(cleaned) < 5:
        return False
    # Reject if all words are stopwords
    if all(w.lower() in _STOPWORDS for w in cleaned.split()):
        return False
    # Reject known noise phrases
    if cleaned.lower() in KP_STOP:
        return False
    # Reject phrases that start with a noise word
    first = cleaned.split()[0].lower()
    if first in {"using","via","with","from","for","and","or","the","a","an"}:
        return False
    # Reject very long phrases (likely sentence fragments)
    if len(cleaned.split()) > 5:
        return False
    return True


class KeyphraseExtractor:

    def extract(self, text: str, top_n: int = 10) -> list:
        y_items = _yake(text)
        t_items = _tfidf_ngrams(text)

        if not y_items and not t_items:
            tokens = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
            tokens = [t for t in tokens if t not in _STOPWORDS and t not in KP_STOP]
            return [w for w, _ in Counter(tokens).most_common(top_n)]

        yn = dict(_normalize(y_items))
        tn = dict(_normalize(t_items))
        all_p = set(yn) | set(tn)

        combined = {}
        for p in all_p:
            cleaned = _clean_phrase(p)
            if not _is_valid_keyphrase(cleaned):
                continue
            # Position bonus: earlier = more relevant
            idx = text.lower().find(p.lower())
            pos = max(0.2 - (idx / max(len(text), 1)) * 2, 0.0) if idx >= 0 else 0.0
            score = 0.5 * yn.get(p, 0) + 0.4 * tn.get(p, 0) + 0.1 * pos
            # Use cleaned phrase as key (deduplicate)
            ck = cleaned.lower()
            if ck not in combined or combined[ck][1] < score:
                combined[ck] = (cleaned, score)

        ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in ranked][:top_n]