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
    "using","video","senior","junior","lead","manager","engineer","scientist",
    "analyst","developer","designer","intern","candidate","professional",
    "specialist","experience","skills","education","projects","work",
    "role","team","years","data","linkedin","kaggle","stream","youtube",
    "github","gahub","gitlab","langcham","langchain","information","please",
    "contact","looking","seeking","motivated","passionate","dedicated",
    "creative","responsible","overview","objective","summary","profile",
    "reference","available","request","currently","previous","following",
    "including","related","various","different","multiple","several",
    # Job title phrases (exact and partial) — should not appear as skills/keyphrases
    "data scientist","machine learning engineer","software engineer",
    "data analyst","product manager","project manager","full stack",
    "front end","back end","frontend developer","backend developer",
    "senior data scientist","junior data scientist","associate data",
    "linkedin data scientist","linkedin machine","linkedin profile",
    "linkedin data","natural language","language processing",
    "experience stealth","full stack machine","stealth startup",
    "kaggle master","language","natural",
}
 
 
def _clean_phrase(phrase: str) -> str:
    """Strip leading/trailing stopwords from a keyphrase."""
    words = phrase.split()
    while words and words[0].lower() in _STOPWORDS:
        words = words[1:]
    while words and words[-1].lower() in _STOPWORDS:
        words = words[:-1]
    return " ".join(words)
 
 
# ── KeyBERT (semantic) ───────────────────────────────────────────────────────
 
_keybert_model = None
 
 
def _get_keybert():
    global _keybert_model
    if _keybert_model is None:
        try:
            from keybert import KeyBERT
            from sentence_transformers import SentenceTransformer
            # Reuse the same lightweight embedding model used elsewhere
            # in the pipeline to avoid loading a second one into memory.
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            _keybert_model = KeyBERT(model=st_model)
        except Exception as e:
            logger.warning(f"KeyBERT load failed: {e}")
    return _keybert_model
 
 
def _keybert_extract(text: str) -> list:
    kb = _get_keybert()
    if kb is None:
        return []
    try:
        kws = kb.extract_keywords(
            text[:6000],
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,        # Maximal Marginal Relevance -> less redundancy
            diversity=0.6,
            top_n=20,
        )
        return kws  # list of (phrase, score)
    except Exception as e:
        logger.warning(f"KeyBERT extraction failed: {e}")
        return []
 
 
# ── Semantic near-duplicate removal ─────────────────────────────────────────
 
def _semantic_dedup(ranked_phrases: list, top_n: int, threshold: float = 0.82) -> list:
    """
    Greedy dedup: keep the highest-scored phrase in each near-duplicate
    cluster (cosine similarity >= threshold), instead of only removing
    exact lowercase string matches.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        phrases = [p for p, _ in ranked_phrases]
        if not phrases:
            return []
        embeddings = model.encode(phrases, normalize_embeddings=True)
 
        kept, kept_idx = [], []
        for i, phrase in enumerate(phrases):
            is_dup = False
            for j in kept_idx:
                sim = float(util.cos_sim(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(phrase)
                kept_idx.append(i)
            if len(kept) >= top_n:
                break
        return kept
    except Exception as e:
        logger.warning(f"Semantic dedup failed, falling back to raw order: {e}")
        return [p for p, _ in ranked_phrases][:top_n]
 
 
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
        k_items = _keybert_extract(text)
 
        if not y_items and not t_items and not k_items:
            tokens = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
            tokens = [t for t in tokens if t not in _STOPWORDS and t not in KP_STOP]
            return [w for w, _ in Counter(tokens).most_common(top_n)]
 
        yn = dict(_normalize(y_items))
        tn = dict(_normalize(t_items))
        kn = dict(_normalize(k_items))
        all_p = set(yn) | set(tn) | set(kn)
 
        combined = {}
        for p in all_p:
            cleaned = _clean_phrase(p)
            if not _is_valid_keyphrase(cleaned):
                continue
            # Position bonus: earlier = more relevant
            idx = text.lower().find(p.lower())
            pos = max(0.2 - (idx / max(len(text), 1)) * 2, 0.0) if idx >= 0 else 0.0
            score = (0.35 * yn.get(p, 0) + 0.30 * tn.get(p, 0)
                     + 0.25 * kn.get(p, 0) + 0.10 * pos)
            # Use cleaned phrase as key (exact-match dedup pass)
            ck = cleaned.lower()
            if ck not in combined or combined[ck][1] < score:
                combined[ck] = (cleaned, score)
 
        ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        # Semantic dedup catches near-duplicates the exact-match pass missed
        # (e.g. "data analysis" vs "analyzing data").
        return _semantic_dedup(ranked, top_n)