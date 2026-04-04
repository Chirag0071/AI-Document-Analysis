"""
utils/summarizer.py — AI Summarisation.

Priority:
  1. Groq API (llama-3.3-70b-versatile) — free, fast, excellent quality
  2. TF-IDF TextRank extractive — always works, zero dependencies
"""

import os
import re
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ── Extractive TextRank (fallback, zero dependencies) ─────────────────────────

def _tokenize(text):
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


def _extractive_summary(text: str, n: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    if not sentences:
        return text[:400]
    if len(sentences) <= n:
        return " ".join(sentences)

    N = len(sentences)
    df = defaultdict(int)
    tf_list = []
    for s in sentences:
        tokens = _tokenize(s)
        freq = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        tf_list.append(freq)
        for t in set(tokens):
            df[t] += 1

    idf = {t: math.log((N + 1) / (df[t] + 1)) for t in df}
    scores = []
    for freq in tf_list:
        total = sum(freq.values()) or 1
        score = sum((freq[t] / total) * idf.get(t, 0) for t in freq)
        scores.append(score)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = sorted([i for i, _ in ranked[:n]])
    return " ".join(sentences[i] for i in top)


# ── Groq API ──────────────────────────────────────────────────────────────────

def _groq_summary(text: str) -> str:
    try:
        from groq import Groq

        key = os.getenv("GROQ_API_KEY")
        if not key:
            logger.warning("GROQ_API_KEY not set in .env")
            return ""

        client = Groq(api_key=key)

        prompt = f"""Summarise the following document in 2-4 sentences (max 100 words).
The summary MUST:
- Identify the document type (invoice, report, resume, article, contract, incident report, etc.)
- Name key participants (people, companies, institutions)
- Include the most important facts: amounts, dates, locations if present
- Use objective, third-person language

Document:
\"\"\"
{text[:6000]}
\"\"\"

Respond with ONLY the summary paragraph. No preamble, no labels, no extra text."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise document summarisation expert. "
                        "Always respond with only a single concise summary paragraph. "
                        "Never add labels, preamble, or extra commentary."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.3,
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        logger.warning(f"Groq API failed: {e}")
        return ""


# ── Public Interface ──────────────────────────────────────────────────────────

class Summarizer:
    def summarize(self, text: str) -> str:
        # 1. Try Groq (Llama 3.3 70B)
        s = _groq_summary(text)
        if s and len(s) > 30:
            return s

        # 2. Extractive TF-IDF TextRank fallback
        logger.info("Using extractive fallback for summarisation")
        return _extractive_summary(text)