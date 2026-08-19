import re
import logging
import random
 
logger = logging.getLogger(__name__)
 
LABELS = [
    "Invoice / Receipt",
    "Resume / CV",
    "Report / Article",
    "Contract / Agreement",
    "Incident Report",
    "Financial Statement",
    "News Article",
    "Academic / Research",
    "Legal Document",
    "General / Other",
]
 
_SEEDS = {
    "Invoice / Receipt": [
        "invoice number date due amount total tax gst hsn payable bill to ship to",
        "receipt payment received amount paid transaction id order number",
        "subtotal discount cgst sgst igst total amount rupees invoice date",
        "billing address shipping address quantity unit price line item total",
        "proforma invoice quotation purchase order vendor supplier buyer seller",
    ],
    "Resume / CV": [
        "resume curriculum vitae objective summary experience education skills",
        "work experience employment history job title company name location dates",
        "bachelor master degree university college gpa cgpa graduated honors",
        "technical skills programming languages frameworks certifications projects",
        "contact email phone linkedin github portfolio achievements awards intern",
    ],
    "Report / Article": [
        "executive summary introduction background methodology findings conclusion",
        "analysis results data statistics percentage growth revenue profit loss",
        "quarterly annual report performance metrics kpi indicators trends growth",
        "industry analysis market share competitive landscape recommendations",
        "technology innovation development research investment infrastructure AI",
    ],
    "Contract / Agreement": [
        "agreement contract parties hereby terms conditions obligations covenants",
        "whereas party first second agrees obligations warranties representations",
        "confidentiality non-disclosure intellectual property rights license grant",
        "termination breach indemnification liability governing law jurisdiction",
        "effective date signatures witnessed executed parties obligations binding",
    ],
    "Incident Report": [
        "incident report breach unauthorized access security vulnerability attack",
        "cybersecurity data breach affected systems investigation response team",
        "incident detected reported escalated resolved root cause analysis",
        "affected users systems networks data compromised security measures taken",
        "incident timeline response containment remediation lessons learned review",
    ],
    "Financial Statement": [
        "balance sheet assets liabilities equity revenue expenses profit loss",
        "income statement cash flow statement retained earnings dividends paid",
        "total revenue operating income ebitda net income earnings per share",
        "accounts receivable payable inventory fixed assets depreciation amort",
        "financial year quarter audit auditor certified public accountant board",
    ],
    "News Article": [
        "reported according sources officials government statement announced said",
        "press release media spokesperson confirmed denied latest development update",
        "breaking news coverage journalist published newspaper magazine editorial",
        "election result policy decision economic impact social political reform",
        "told reported noted warned called asked responded officials authorities",
    ],
    "Academic / Research": [
        "abstract introduction literature review methodology results discussion",
        "hypothesis experiment data collection analysis statistical significance",
        "journal paper citation references doi published peer reviewed conference",
        "figure table appendix supplementary material dataset sample size n study",
        "proposed model algorithm evaluation benchmark performance accuracy recall",
    ],
    "Legal Document": [
        "court judge plaintiff defendant petitioner respondent affidavit sworn",
        "whereas henceforth notwithstanding pursuant herein heretofore therein",
        "order decree judgment conviction sentence appeal tribunal bench bar",
        "section clause subsection act statute regulation compliance penalty fine",
        "legal notice summons writ petition case number filed registered sealed",
    ],
    "General / Other": [
        "document text information content general purpose various topics note",
        "letter memo communication correspondence official informal personal",
        "notice circular announcement information update guidelines policy rules",
        "form application request submission registration details required fields",
        "manual guide instructions procedure steps process workflow standard ops",
    ],
}
 
 
# ── Sentence-Embedding Model (lazy singleton, shared across calls) ─────────────
 
_embed_model = None
_label_embeddings = None  # np.ndarray, one centroid per label
 
 
def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Sentence-transformer load failed: {e}")
    return _embed_model
 
 
def _build_label_embeddings():
    """Embed each label's seed phrases once, average into a centroid."""
    global _label_embeddings
    model = _get_embed_model()
    if model is None:
        return None
    if _label_embeddings is not None:
        return _label_embeddings
    try:
        import numpy as np
        centroids = []
        for label in LABELS:
            seeds = _SEEDS[label]
            vecs = model.encode(seeds, normalize_embeddings=True)
            centroids.append(np.mean(vecs, axis=0))
        _label_embeddings = np.vstack(centroids)
        return _label_embeddings
    except Exception as e:
        logger.warning(f"Building label embeddings failed: {e}")
        return None
 
 
def _embedding_predict(text: str):
    """Return (probability-vector-over-LABELS) via cosine similarity, or None."""
    model = _get_embed_model()
    centroids = _build_label_embeddings()
    if model is None or centroids is None:
        return None
    try:
        import numpy as np
        vec = model.encode([text[:2000]], normalize_embeddings=True)[0]
        sims = centroids @ vec  # cosine similarity (both normalized)
        # Softmax-style conversion to a probability distribution
        exp = np.exp((sims - sims.max()) * 8)  # temperature sharpens confident calls
        probs = exp / exp.sum()
        return probs
    except Exception as e:
        logger.warning(f"Embedding predict failed: {e}")
        return None
 
 
def _training_data():
    random.seed(42)
    X, y = [], []
    for label, seeds in _SEEDS.items():
        idx = LABELS.index(label)
        for seed in seeds:
            X.append(seed)
            y.append(idx)
            words = seed.split()
            for _ in range(4):
                random.shuffle(words)
                X.append(" ".join(words))
                y.append(idx)
    return X, y
 
 
class DocumentClassifier:
 
    def __init__(self):
        self._vec = None
        self._rf  = None
        self._build()
 
    def _build(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
 
            X_texts, y = _training_data()
 
            self._vec = TfidfVectorizer(
                max_features=800,
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
            )
            X = self._vec.fit_transform(X_texts)
 
            self._rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self._rf.fit(X, np.array(y))
            logger.info("DocumentClassifier: RandomForest trained OK")
        except Exception as e:
            logger.warning(f"DocumentClassifier build failed: {e}")
 
    def _keyword_fallback(self, text: str) -> str:
        tl = text.lower()
        kw = {
            "Invoice / Receipt":    ["invoice","receipt","bill","gst","payable","amount due"],
            "Resume / CV":          ["resume","cv","curriculum vitae","experience","skills"],
            "Contract / Agreement": ["agreement","contract","hereby","parties","terms"],
            "Incident Report":      ["incident","breach","attack","vulnerability","unauthorized"],
            "Financial Statement":  ["balance sheet","income statement","assets","liabilities"],
            "News Article":         ["according to","reported","officials said","announced"],
            "Academic / Research":  ["abstract","hypothesis","methodology","peer reviewed"],
            "Legal Document":       ["court","plaintiff","defendant","judgment","pursuant"],
            "Report / Article":     ["executive summary","findings","recommendations","analysis"],
        }
        scores = {l: sum(1 for k in kws if k in tl) for l, kws in kw.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "General / Other"
 
    def classify(self, text: str) -> str:
        import numpy as np
 
        rf_proba = None
        if self._rf is not None and self._vec is not None:
            try:
                X = self._vec.transform([text[:5000]])
                rf_proba = self._rf.predict_proba(X)[0]
            except Exception as e:
                logger.warning(f"RF classify failed: {e}")
 
        emb_proba = _embedding_predict(text)
 
        if rf_proba is None and emb_proba is None:
            return self._keyword_fallback(text)
 
        if rf_proba is not None and emb_proba is not None:
            # Ensemble: semantic model gets slightly more weight since it
            # generalizes better to phrasing the lexical model never saw.
            combined = 0.6 * emb_proba + 0.4 * rf_proba
        else:
            combined = emb_proba if emb_proba is not None else rf_proba
 
        if max(combined) < 0.30:
            return "General / Other"
        return LABELS[int(np.argmax(combined))]
 