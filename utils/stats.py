"""
utils/stats.py — Document statistics and readability metrics.
"""

import re
import math
import logging

logger = logging.getLogger(__name__)

_VOWELS = set("aeiouAEIOU")


def _syllables(word: str) -> int:
    word = word.lower().strip(".,;:!?")
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:es|ed|e)$", "", word)
    count = sum(1 for i, c in enumerate(word)
                if c in _VOWELS and (i == 0 or word[i-1] not in _VOWELS))
    return max(count, 1)


class DocumentStats:
    def compute(self, raw_text: str, clean_text: str) -> dict:
        words = re.findall(r"\b[a-zA-Z\']{1,}\b", clean_text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]

        nw = len(words)
        ns = max(len(sentences), 1)
        np_ = max(len(paragraphs), 1)

        avg_sl = round(nw / ns, 1)
        avg_wl = round(sum(len(w) for w in words) / max(nw, 1), 1)

        sylls = [_syllables(w) for w in words]
        avg_sy = sum(sylls) / max(nw, 1)
        pct_cx = (sum(1 for s in sylls if s >= 3) / max(nw, 1)) * 100

        fre  = round(max(0, min(100, 206.835 - 1.015*avg_sl - 84.6*avg_sy)), 1)
        fkgl = round(0.39*avg_sl + 11.8*avg_sy - 15.59, 1)
        fog  = round(0.4*(avg_sl + pct_cx), 1)

        ttr  = round(len(set(w.lower() for w in words)) / max(nw, 1), 3)

        lang = "en"
        try:
            from langdetect import detect
            if nw > 20:
                lang = detect(clean_text[:2000])
        except Exception:
            pass

        return {
            "word_count":           nw,
            "sentence_count":       ns,
            "paragraph_count":      np_,
            "character_count":      len(clean_text),
            "avg_sentence_length":  avg_sl,
            "avg_word_length":      avg_wl,
            "lexical_diversity":    ttr,
            "reading_time_minutes": round(nw / 200, 1),
            "language":             lang,
            "readability": {
                "flesch_reading_ease":  fre,
                "flesch_kincaid_grade": fkgl,
                "gunning_fog_index":    fog,
                "interpretation": (
                    "Very Easy"      if fre >= 90 else
                    "Easy"           if fre >= 70 else
                    "Standard"       if fre >= 50 else
                    "Difficult"      if fre >= 30 else
                    "Very Difficult"
                ),
            },
        }
