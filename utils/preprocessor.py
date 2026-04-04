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

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFC", text)

        for bad, good in self._UNICODE_FIXES.items():
            text = text.replace(bad, good)

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
