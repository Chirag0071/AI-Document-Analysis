"""
utils/summarizer.py — AI Summarisation + Full Analysis via Groq.

groq_full_analysis() — ONE call returns summary + entities + sentiment as JSON
Summarizer.summarize() — fallback extractive summary if Groq fails
"""

import os
import re
import json
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ── Groq Full Analysis (summary + entities + sentiment in ONE call) ────────────

def groq_full_analysis(text: str) -> dict:
    """
    Single Groq API call returning summary, entities AND sentiment together.
    Returns empty dict if Groq unavailable.
    """
    try:
        from groq import Groq

        key = os.getenv("GROQ_API_KEY")
        if not key:
            logger.warning("GROQ_API_KEY not set in .env")
            return {}

        client = Groq(api_key=key)

        prompt = f"""You are a document analysis expert. Analyze the document below and return ONLY a valid JSON object. No explanation, no markdown, no code blocks — pure JSON only.

The JSON must have exactly these fields:
{{
  "summary": "2-4 sentence summary. Must identify: (1) document type, (2) key people/companies, (3) important facts like amounts, dates, locations.",
  "sentiment": "Positive or Negative or Neutral",
  "entities": {{
    "names": ["full person names only, e.g. John Smith"],
    "dates": ["specific dates only, e.g. 10 March 2026"],
    "organizations": ["company/org names only, e.g. ABC Pvt Ltd"],
    "locations": ["city/country names only, e.g. New York"],
    "amounts": ["monetary amounts only with currency symbol, e.g. Rs.10000 or $5000"]
  }}
}}

Rules:
- sentiment must be exactly one of: Positive, Negative, Neutral
- names must be real person names only (not job titles, not companies)
- amounts must contain a number and currency (not just "rs" or "rupees")
- dates must be specific (not "past few years")
- organizations must be proper company/institution names only
- if a field has no valid values, use an empty list []
- Return ONLY the JSON, nothing else

Document:
\"\"\"
{text[:5000]}
\"\"\"
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document analysis API. You ONLY return valid JSON. "
                        "Never add explanation, markdown formatting, or code blocks. "
                        "Return pure JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        # Clean any accidental markdown
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        raw = raw.strip()

        # Find JSON object in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        result = json.loads(raw)

        # Validate structure
        if not isinstance(result.get("summary"), str):
            result["summary"] = ""
        if result.get("sentiment") not in ("Positive", "Negative", "Neutral"):
            result["sentiment"] = ""
        if not isinstance(result.get("entities"), dict):
            result["entities"] = {}

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Groq returned invalid JSON: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Groq full analysis failed: {e}")
        return {}


# ── Extractive TextRank Fallback ──────────────────────────────────────────────

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


# ── Public Interface ──────────────────────────────────────────────────────────

class Summarizer:
    def summarize(self, text: str) -> str:
        """Fallback extractive summary when Groq is unavailable."""
        return _extractive_summary(text)