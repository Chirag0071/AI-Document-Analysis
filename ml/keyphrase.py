"""
ml/keyphrase.py — Key-phrase extraction.

YAKE (unsupervised statistical) + TF-IDF n-gram hybrid reranker.
Falls back to frequency-based extraction if YAKE unavailable.
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


def _yake(text: str) -> list:
    try:
        import yake
        ex = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7,
                                   dedupFunc="seqm", windowsSize=1, top=20)
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
              for p, c in tf.items() if len(p) >= 4}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:20]


def _normalize(items):
    if not items:
        return []
    mn, mx = min(s for _, s in items), max(s for _, s in items)
    rng = mx - mn or 1
    return [(p, (s-mn)/rng) for p, s in items]


class KeyphraseExtractor:

    def extract(self, text: str, top_n: int = 10) -> list:
        y_items = _yake(text)
        t_items = _tfidf_ngrams(text)

        if not y_items and not t_items:
            tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            tokens = [t for t in tokens if t not in _STOPWORDS]
            return [w for w, _ in Counter(tokens).most_common(top_n)]

        yn = dict(_normalize(y_items))
        tn = dict(_normalize(t_items))
        all_p = set(yn) | set(tn)

        combined = {}
        for p in all_p:
            # Position bonus: earlier = more relevant
            idx = text.lower().find(p.lower())
            pos = max(0.2 - (idx / max(len(text), 1)) * 2, 0.0) if idx >= 0 else 0.0
            combined[p] = 0.5*yn.get(p, 0) + 0.4*tn.get(p, 0) + 0.1*pos

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked
                if len(p) >= 4 and not all(w in _STOPWORDS for w in p.split())][:top_n]
