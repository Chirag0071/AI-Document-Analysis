"""
pipeline/processor.py — Master orchestrator.

Strategy:
  1. Extract text (PDF/DOCX/Image)
  2. Try ONE Groq call → returns summary + entities + sentiment together
  3. If Groq fails → fall back to individual ML/NLP modules
  4. Always merge Groq + spaCy entities for best coverage
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.extractor import TextExtractor
from utils.preprocessor import TextPreprocessor
from utils.summarizer import Summarizer, groq_full_analysis
from utils.entity import EntityExtractor
from utils.sentiment import SentimentAnalyzer
from utils.stats import DocumentStats
from ml.classifier import DocumentClassifier
from ml.keyphrase import KeyphraseExtractor

logger = logging.getLogger(__name__)


def _merge_entities(groq_ents: dict, spacy_ents: dict) -> dict:
    """
    Merge entities from Groq LLM and spaCy NER.
    Deduplicates case-insensitively, keeps all unique values from both.
    """
    keys = ["names", "dates", "organizations", "locations",
            "amounts", "percentages", "emails", "phones", "urls"]
    result = {}
    for key in keys:
        g = groq_ents.get(key, [])
        s = spacy_ents.get(key, [])
        combined = list(g) + list(s)
        # Deduplicate case-insensitively
        seen = {}
        for item in combined:
            k = str(item).strip().lower()
            if k and len(k) > 1 and k not in seen:
                seen[k] = str(item).strip()
        result[key] = list(seen.values())
    return result


def _clean_groq_entities(ents: dict) -> dict:
    """Remove obvious garbage from Groq entity output."""
    import re
    cleaned = {}
    for key, values in ents.items():
        if not isinstance(values, list):
            cleaned[key] = []
            continue
        clean = []
        for v in values:
            v = str(v).strip()
            if not v or len(v) < 2:
                continue
            # Skip amounts without digits
            if key == "amounts" and not re.search(r"\d{2,}", v):
                continue
            # Skip very long entries (likely sentences not entities)
            if len(v.split()) > 6:
                continue
            clean.append(v)
        cleaned[key] = clean
    return cleaned


class DocumentProcessor:
    """Loads all models once at startup; process() called per request."""

    def __init__(self):
        self._extractor    = TextExtractor()
        self._preprocessor = TextPreprocessor()
        self._summarizer   = Summarizer()
        self._ner          = EntityExtractor()
        self._sentiment    = SentimentAnalyzer()
        self._stats        = DocumentStats()
        self._classifier   = DocumentClassifier()
        self._keyphrase    = KeyphraseExtractor()

    def process(self, file_bytes: bytes, file_type: str, file_name: str) -> dict:

        # ── 1. Extract raw text ────────────────────────────────────────────
        raw_text = self._extractor.extract(file_bytes, file_type)
        if not raw_text or not raw_text.strip():
            raise ValueError("No text could be extracted from the document.")

        # ── 2. Clean text ──────────────────────────────────────────────────
        clean_text = self._preprocessor.clean(raw_text)

        # ── 3. Try Groq full analysis (summary + entities + sentiment) ─────
        groq_result = groq_full_analysis(clean_text)

        # ── 4. Extract with spaCy + regex (always runs) ───────────────────
        spacy_entities = self._ner.extract(clean_text)

        # ── 5. Decide summary ─────────────────────────────────────────────
        if groq_result.get("summary") and len(groq_result["summary"]) > 30:
            summary = groq_result["summary"]
        else:
            summary = self._summarizer.summarize(clean_text)

        # ── 6. Decide sentiment ───────────────────────────────────────────
        if groq_result.get("sentiment") in ("Positive", "Negative", "Neutral"):
            sentiment_label = groq_result["sentiment"]
            # Still get scores from ML ensemble
            ml_sentiment = self._sentiment.analyze(clean_text)
            # Trust Groq label, use ML scores
            sentiment_scores = ml_sentiment["scores"]
            # Align scores with Groq label
            if sentiment_label == "Negative":
                sentiment_scores = {"Positive": 0.05, "Neutral": 0.15, "Negative": 0.80}
            elif sentiment_label == "Positive":
                sentiment_scores = {"Positive": 0.80, "Neutral": 0.15, "Negative": 0.05}
            else:
                sentiment_scores = {"Positive": 0.15, "Neutral": 0.70, "Negative": 0.15}
        else:
            ml_sentiment = self._sentiment.analyze(clean_text)
            sentiment_label = ml_sentiment["label"]
            sentiment_scores = ml_sentiment["scores"]

        # ── 7. Merge entities (Groq + spaCy for best coverage) ────────────
        groq_ents = _clean_groq_entities(groq_result.get("entities", {}))
        merged_entities = _merge_entities(groq_ents, spacy_entities)

        # ── 8. Document classification ────────────────────────────────────
        doc_type = self._classifier.classify(clean_text)

        # ── 9. Key-phrases ────────────────────────────────────────────────
        keyphrases = self._keyphrase.extract(clean_text)

        # ── 10. Statistics ────────────────────────────────────────────────
        stats = self._stats.compute(raw_text, clean_text)

        return {
            "status":           "success",
            "fileName":         file_name,
            "summary":          summary,
            "entities":         merged_entities,
            "sentiment":        sentiment_label,
            "sentiment_scores": sentiment_scores,
            "document_type":    doc_type,
            "key_phrases":      keyphrases,
            "document_stats":   stats,
        }