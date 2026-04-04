"""
utils/extractor.py — Multi-format text extraction.

PDF   → pdfplumber (primary) → PyMuPDF (fallback) → OCR (last resort)
DOCX  → python-docx (paragraphs + tables + headers/footers)
Image → OpenCV preprocessing → Tesseract OCR (PSM 6 + PSM 3)
"""

import io
import logging
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
logger = logging.getLogger(__name__)


class TextExtractor:

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _pdf_pdfplumber(self, file_bytes: bytes) -> str:
        import pdfplumber
        texts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t and t.strip():
                    texts.append(t.strip())
                else:
                    words = page.extract_words()
                    if words:
                        texts.append(" ".join(w["text"] for w in words))
        return "\n\n".join(texts)

    def _pdf_pymupdf(self, file_bytes: bytes) -> str:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        parts = [page.get_text("text").strip() for page in doc if page.get_text("text").strip()]
        return "\n\n".join(parts)

    def _pdf_ocr(self, file_bytes: bytes) -> str:
        import fitz
        import pytesseract
        from PIL import Image
        import numpy as np
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page in doc:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img = Image.fromarray(
                __import__("numpy").frombuffer(pix.samples, dtype=__import__("numpy").uint8
                ).reshape(pix.height, pix.width)
            )
            t = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            if t.strip():
                texts.append(t.strip())
        return "\n\n".join(texts)

    def _extract_pdf(self, file_bytes: bytes) -> str:
        text = ""
        try:
            text = self._pdf_pdfplumber(file_bytes)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        if len(text.strip()) < 100:
            try:
                text = self._pdf_pymupdf(file_bytes)
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}")
        if len(text.strip()) < 50:
            try:
                text = self._pdf_ocr(file_bytes)
            except Exception as e:
                logger.warning(f"PDF OCR failed: {e}")
        return text

    # ── DOCX ──────────────────────────────────────────────────────────────────

    def _extract_docx(self, file_bytes: bytes) -> str:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        parts = []

        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                if para.style.name.startswith("Heading"):
                    parts.append(f"[HEADING] {t}")
                else:
                    parts.append(t)

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        try:
            for section in doc.sections:
                for para in section.header.paragraphs:
                    if para.text.strip():
                        parts.append(f"[HEADER] {para.text.strip()}")
                for para in section.footer.paragraphs:
                    if para.text.strip():
                        parts.append(f"[FOOTER] {para.text.strip()}")
        except Exception:
            pass

        return "\n".join(parts)

    # ── IMAGE ─────────────────────────────────────────────────────────────────

    def _preprocess_image(self, file_bytes: bytes):
        import cv2
        import numpy as np
        from PIL import Image

        nparr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]
        if max(h, w) < 1500:
            scale = 1500 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 8
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(thresh)

    def _extract_image(self, file_bytes: bytes) -> str:
        import pytesseract
        pil_img = self._preprocess_image(file_bytes)
        t1 = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 6").strip()
        t2 = ""
        if len(t1) < 200:
            t2 = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 3").strip()
        return max([t1, t2], key=len)

    # ── Router ────────────────────────────────────────────────────────────────

    def extract(self, file_bytes: bytes, file_type: str) -> str:
        ft = file_type.lower().strip()
        if ft == "pdf":
            return self._extract_pdf(file_bytes)
        elif ft == "docx":
            return self._extract_docx(file_bytes)
        elif ft == "image":
            return self._extract_image(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
