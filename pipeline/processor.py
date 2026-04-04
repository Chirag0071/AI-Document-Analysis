"""
pipeline/processor.py — Master orchestrator.
Runs all 8 stages and assembles the final JSON response.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.extractor import TextExtractor
from utils.preprocessor import TextPreprocessor
from utils.summarizer import Summarizer
from utils.entity import EntityExtractor
from utils.sentiment import SentimentAnalyzer
from utils.stats import DocumentStats
from ml.classifier import DocumentClassifier
from ml.keyphrase import KeyphraseExtractor


class DocumentProcessor:
    """Loads all models once at startup; process() is called per request."""

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
        # 1. Extract raw text
        raw_text = self._extractor.extract(file_bytes, file_type)
        if not raw_text or not raw_text.strip():
            raise ValueError("No text could be extracted from the document.")

        # 2. Clean text
        clean_text = self._preprocessor.clean(raw_text)

        # 3. Summarise
        summary = self._summarizer.summarize(clean_text)

        # 4. Named Entity Recognition
        entities = self._ner.extract(clean_text)

        # 5. Sentiment
        sentiment_result = self._sentiment.analyze(clean_text)

        # 6. Document Classification
        doc_type = self._classifier.classify(clean_text)

        # 7. Key-phrases
        keyphrases = self._keyphrase.extract(clean_text)

        # 8. Statistics
        stats = self._stats.compute(raw_text, clean_text)

        return {
            "status":            "success",
            "fileName":          file_name,
            "summary":           summary,
            "entities":          entities,
            "sentiment":         sentiment_result["label"],
            "sentiment_scores":  sentiment_result["scores"],
            "document_type":     doc_type,
            "key_phrases":       keyphrases,
            "document_stats":    stats,
        }
