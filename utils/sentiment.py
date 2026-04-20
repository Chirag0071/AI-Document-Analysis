"""
utils/sentiment.py — 4-model ensemble sentiment analysis.

"""

import re
import logging

logger = logging.getLogger(__name__)


# ── VADER ─────────────────────────────────────────────────────────────────────

def _vader_scores(text: str) -> dict:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)
                  if text[i:i+4000].strip()]
        if not chunks:
            return {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}
        all_s = [analyzer.polarity_scores(c) for c in chunks]
        return {k: sum(s[k] for s in all_s) / len(all_s)
                for k in ("compound", "pos", "neg", "neu")}
    except Exception:
        return {"compound": 0.0, "pos": 0.33, "neg": 0.33, "neu": 0.34}


# ── TextBlob ──────────────────────────────────────────────────────────────────

def _textblob_scores(text: str) -> dict:
    try:
        from textblob import TextBlob
        b = TextBlob(text[:8000])
        return {
            "polarity": b.sentiment.polarity,
            "subjectivity": b.sentiment.subjectivity
        }
    except Exception:
        return {"polarity": 0.0, "subjectivity": 0.5}


# ── Transformer ───────────────────────────────────────────────────────────────

_trans_pipe = None

def _transformer_score(text: str) -> float:
    global _trans_pipe
    try:
        if _trans_pipe is None:
            from transformers import pipeline as hf_pipeline
            for mid in [
                "ProsusAI/finbert",
                "distilbert-base-uncased-finetuned-sst-2-english"
            ]:
                try:
                    _trans_pipe = hf_pipeline(
                        "text-classification", model=mid,
                        device=-1, truncation=True, max_length=512
                    )
                    break
                except Exception:
                    continue
        if not _trans_pipe:
            return 0.0
        r = _trans_pipe(text[:1500])[0]
        label = r["label"].lower()
        score = r["score"]
        if "positive" in label:
            return score
        elif "negative" in label:
            return -score
        return 0.0
    except Exception:
        return 0.0


# ── Sentence-level features ───────────────────────────────────────────────────

def _sent_features(text: str) -> dict:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text[:6000])
                 if len(s.strip()) > 10]
        if not sents:
            return {"avg": 0.0, "prop_pos": 0.33, "prop_neg": 0.33}
        comps = [analyzer.polarity_scores(s)["compound"] for s in sents]
        n = len(comps)
        return {
            "avg":      sum(comps) / n,
            "prop_pos": sum(1 for c in comps if c >= 0.05) / n,
            "prop_neg": sum(1 for c in comps if c <= -0.05) / n,
        }
    except Exception:
        return {"avg": 0.0, "prop_pos": 0.33, "prop_neg": 0.33}


# ── Decision Tree Meta-Classifier ─────────────────────────────────────────────

class _DTMeta:
    def __init__(self):
        self._dt = None
        try:
            from sklearn.tree import DecisionTreeClassifier
            import numpy as np

            X = np.array([
                [ 0.80,0.45,0.05,0.50, 0.60,0.70, 0.75,0.80,0.05, 0.85],
                [ 0.70,0.40,0.05,0.55, 0.50,0.60, 0.65,0.70,0.10, 0.75],
                [ 0.60,0.35,0.08,0.57, 0.45,0.55, 0.55,0.65,0.15, 0.65],
                [ 0.50,0.30,0.10,0.60, 0.35,0.50, 0.45,0.60,0.20, 0.55],
                [-0.80,0.05,0.45,0.50,-0.60,0.65,-0.75,0.05,0.80,-0.85],
                [-0.70,0.08,0.40,0.52,-0.55,0.60,-0.65,0.10,0.75,-0.75],
                [-0.60,0.10,0.35,0.55,-0.45,0.55,-0.55,0.15,0.65,-0.65],
                [-0.50,0.12,0.30,0.58,-0.35,0.50,-0.45,0.20,0.60,-0.55],
                [ 0.02,0.15,0.13,0.72, 0.05,0.30, 0.01,0.35,0.30, 0.05],
                [ 0.01,0.14,0.14,0.72, 0.02,0.25,-0.01,0.30,0.32, 0.02],
                [-0.02,0.13,0.15,0.72,-0.05,0.28,-0.02,0.28,0.35,-0.03],
                [ 0.00,0.12,0.12,0.76, 0.00,0.20, 0.00,0.33,0.33, 0.00],
                [ 0.15,0.25,0.20,0.55, 0.10,0.60, 0.12,0.45,0.35, 0.10],
                [-0.15,0.20,0.25,0.55,-0.10,0.60,-0.12,0.35,0.45,-0.10],
                [ 0.55,0.38,0.10,0.52, 0.40,0.85, 0.50,0.62,0.18, 0.60],
                [-0.40,0.08,0.25,0.67,-0.30,0.30,-0.38,0.10,0.55,-0.45],
            ], dtype=float)
            y = [2,2,2,2, 0,0,0,0, 1,1,1,1, 1,1, 2,0]

            self._dt = DecisionTreeClassifier(max_depth=8, random_state=42)
            self._dt.fit(X, y)
        except Exception as e:
            logger.warning(f"DT meta-classifier unavailable: {e}")

    def predict(self, feats: list) -> tuple:
        labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        if self._dt is None:
            c = feats[0]
            if c >= 0.05:
                return "Positive", {"Positive": 0.80, "Neutral": 0.15, "Negative": 0.05}
            if c <= -0.05:
                return "Negative", {"Positive": 0.05, "Neutral": 0.15, "Negative": 0.80}
            return "Neutral", {"Positive": 0.10, "Neutral": 0.80, "Negative": 0.10}

        import numpy as np
        X = np.array([feats])
        pred = self._dt.predict(X)[0]
        proba = self._dt.predict_proba(X)[0]
        classes = self._dt.classes_
        scores = {labels[c]: round(float(p), 3) for c, p in zip(classes, proba)}
        return labels[pred], scores


# ── Public Interface ──────────────────────────────────────────────────────────

class SentimentAnalyzer:
    def __init__(self):
        self._meta = _DTMeta()

    def analyze(self, text: str) -> dict:
        vader = _vader_scores(text)
        blob  = _textblob_scores(text)
        sents = _sent_features(text)
        trans = _transformer_score(text)

        feats = [
            vader["compound"], vader["pos"], vader["neg"], vader["neu"],
            blob["polarity"],  blob["subjectivity"],
            sents["avg"], sents["prop_pos"], sents["prop_neg"],
            trans,
        ]

        label, scores = self._meta.predict(feats)

        # Strong agreement override — both VADER and TextBlob agree
        if vader["compound"] <= -0.05 and blob["polarity"] < 0:
            label = "Negative"
            scores = {"Positive": 0.05, "Neutral": 0.15, "Negative": 0.80}
        elif vader["compound"] >= 0.05 and blob["polarity"] > 0:
            label = "Positive"
            scores = {"Positive": 0.80, "Neutral": 0.15, "Negative": 0.05}

        return {"label": label, "scores": scores}