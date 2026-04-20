"""
utils/preprocessor.py — Text cleaning and normalization.

"""

import re
import unicodedata


class TextPreprocessor:

    _UNICODE_FIXES = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2022": " ", "\u00a0": " ",
        "\u00ad": "",  "\ufb01": "fi",
        "\ufb02": "fl", "\u0000": "",
    }

    # OCR normalization — fixes common Tesseract mangling.
    # Keys are regex patterns (applied with re.sub, case-sensitive where needed).
    # Values are replacement strings.
    _OCR_NORMALIZE = [
        # Specific OCR typos found in testing
        (r"\bCLCD\b",      "CI/CD"),
        (r"\bCl/CD\b",     "CI/CD"),
        (r"\bCl CD\b",     "CI/CD"),
        (r"\bCI\/C0\b",    "CI/CD"),
        (r"\bFastAP\b",    "FastAPI"),
        (r"\bFastAp\b",    "FastAPI"),
        (r"\bFastApl\b",   "FastAPI"),
        (r"\bPythan\b",    "Python"),
        (r"\bPython3\b",   "Python"),
        (r"\bTensorFIow\b","TensorFlow"),
        (r"\bTensorFlow\b","TensorFlow"),
        (r"\bPyToreh\b",   "PyTorch"),
        (r"\bPyTorch\b",   "PyTorch"),
        (r"\bGitHub\b",    "GitHub"),
        (r"\bGitHiub\b",   "GitHub"),
        (r"\bLinkedln\b",  "LinkedIn"),
        (r"\bLinkedIn\b",  "LinkedIn"),
        (r"\bMongoDB\b",   "MongoDB"),
        (r"\bDockar\b",    "Docker"),
        (r"\bKubernates\b","Kubernetes"),
        (r"\bJavascript\b","JavaScript"),
        (r"\bjavascript\b","JavaScript"),
        (r"\bReactJS\b",   "React.js"),
        (r"\bNextJS\b",    "Next.js"),
        (r"\bNodeJS\b",    "Node.js"),
        (r"\bNodcJS\b",    "Node.js"),
        # Broken hyphenation from PDF extraction
        (r"(\w)-\s*\n\s*(\w)", r"\1\2"),
        # Common OCR character swaps
        (r"\b0racle\b",    "Oracle"),
        (r"\b0penAI\b",    "OpenAI"),
        (r"\bAzurc\b",     "Azure"),
        # Fix common number/letter confusion in tech terms
        (r"\bMachme\b",    "Machine"),
        (r"\bLearnmg\b",   "Learning"),
    ]

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFC", text)

        for bad, good in self._UNICODE_FIXES.items():
            text = text.replace(bad, good)

        # ── OCR normalization ──────────────────────────────────────────────
        for pattern, replacement in self._OCR_NORMALIZE:
            text = re.sub(pattern, replacement, text)

        # Remove extractor structural markers
        text = re.sub(r"\[(HEADING|HEADER|FOOTER)\]\s*", "", text)

        # Fix OCR hyphenation at line breaks
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Remove garbage lines (< 40% alphanumeric)
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            s = line.strip()
            if len(s) < 3:
                clean_lines.append("")
                continue
            alnum = sum(c.isalnum() or c.isspace() for c in s)
            if alnum / max(len(s), 1) >= 0.40:
                clean_lines.append(s)

        text = "\n".join(clean_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        paragraphs = text.split("\n\n")
        paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in paragraphs]
        return "\n\n".join(p for p in paragraphs if p).strip()

    def sentences(self, text: str) -> list:
        protected = re.sub(
            r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.",
            r"\1<DOT>", text
        )
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
        return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]