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
 
 
# ── Transformers (lazy-loaded singletons, reused across calls) ────────────────
 
_finbert_pipe = None
_roberta_pipe = None
 
 
def _get_finbert():
    global _finbert_pipe
    if _finbert_pipe is None:
        try:
            from transformers import pipeline as hf_pipeline
            for mid in [
                "ProsusAI/finbert",
                "distilbert-base-uncased-finetuned-sst-2-english",
            ]:
                try:
                    _finbert_pipe = hf_pipeline(
                        "text-classification", model=mid,
                        device=-1, truncation=True, max_length=512,
                    )
                    break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e}")
    return _finbert_pipe
 
 
def _get_roberta():
    """General-purpose transformer, independent of finance-domain bias."""
    global _roberta_pipe
    if _roberta_pipe is None:
        try:
            from transformers import pipeline as hf_pipeline
            _roberta_pipe = hf_pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1, truncation=True, max_length=512,
            )
        except Exception as e:
            logger.warning(f"RoBERTa load failed: {e}")
    return _roberta_pipe
 
 
def _score_from_result(r: dict) -> float:
    label = r["label"].lower()
    score = r["score"]
    if "pos" in label or label == "label_2":
        return score
    if "neg" in label or label == "label_0":
        return -score
    return 0.0
 
 
def _transformer_scores(text: str) -> dict:
    """Run both transformer models once, return their signed scores."""
    chunk = text[:1500]
    out = {"finbert": 0.0, "roberta": 0.0}
 
    finbert = _get_finbert()
    if finbert:
        try:
            out["finbert"] = _score_from_result(finbert(chunk)[0])
        except Exception:
            pass
 
    roberta = _get_roberta()
    if roberta:
        try:
            out["roberta"] = _score_from_result(roberta(chunk)[0])
        except Exception:
            pass
 
    return out
 
 
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
        trans = _transformer_scores(text)
        trans_avg = (trans["finbert"] + trans["roberta"]) / 2
 
        feats = [
            vader["compound"], vader["pos"], vader["neg"], vader["neu"],
            blob["polarity"],  blob["subjectivity"],
            sents["avg"], sents["prop_pos"], sents["prop_neg"],
            trans_avg,
        ]
 
        label, dt_scores = self._meta.predict(feats)
 
        # Soft-vote blend: combine the DT meta-classifier's probability
        # estimate with a normalized signal from every model in the
        # ensemble, instead of overwriting with fixed numbers. Each
        # model's signed score in [-1, 1] is mapped into Pos/Neu/Neg
        # weight via a simple triangular split, then averaged.
        signals = [vader["compound"], blob["polarity"],
                   trans["finbert"], trans["roberta"]]
        signals = [s for s in signals if s != 0.0] or [0.0]
 
        def _to_probs(s: float) -> dict:
            pos = max(s, 0.0)
            neg = max(-s, 0.0)
            neu = max(1 - abs(s), 0.0)
            total = pos + neg + neu or 1.0
            return {"Positive": pos / total, "Neutral": neu / total,
                    "Negative": neg / total}
 
        ensemble_probs = [_to_probs(s) for s in signals]
        blended = {
            k: sum(p[k] for p in ensemble_probs) / len(ensemble_probs)
            for k in ("Positive", "Neutral", "Negative")
        }
 
        # Final scores: average of the DT classifier's estimate and the
        # ensemble soft-vote (each retains real signal, no hardcoding).
        final_scores = {
            k: round(0.5 * dt_scores.get(k, 0.0) + 0.5 * blended[k], 3)
            for k in ("Positive", "Neutral", "Negative")
        }
        # Normalize to sum to 1 after rounding
        total = sum(final_scores.values()) or 1.0
        final_scores = {k: round(v / total, 3) for k, v in final_scores.items()}
 
        final_label = max(final_scores, key=final_scores.get)
 
        return {"label": final_label, "scores": final_scores}