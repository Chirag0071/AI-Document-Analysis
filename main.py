"""
main.py — AI-Powered Document Analysis API
Run from the HCL/ folder:
    uvicorn main:app --reload
"""

import os
import sys
import base64
import time
import hashlib

# ── Fix import path so pipeline/utils/ml are always found ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from pipeline.processor import DocumentProcessor

load_dotenv()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Powered Document Analysis API",
    description="Extracts text from PDF, DOCX, and images then runs a full NLP + LLM pipeline.",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

processor = DocumentProcessor()
API_KEY = os.getenv("API_KEY", "sk_track2_987654321")


# ── Request Model ──────────────────────────────────────────────────────────────
class DocumentRequest(BaseModel):
    fileName: str
    fileType: str
    fileBase64: str

    @field_validator("fileType")
    @classmethod
    def validate_type(cls, v):
        v = v.lower().strip()
        if v not in {"pdf", "docx", "image"}:
            raise ValueError("fileType must be one of: pdf, docx, image")
        return v


# ── Auth ───────────────────────────────────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(...)):
    provided = hashlib.sha256(x_api_key.encode()).digest()
    expected = hashlib.sha256(API_KEY.encode()).digest()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key.")
    return x_api_key


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "AI Document Analysis API", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/document-analyze")
async def analyze_document(body: DocumentRequest, _: str = Depends(verify_api_key)):
    t0 = time.perf_counter()

    try:
        file_bytes = base64.b64decode(body.fileBase64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    try:
        result = processor.process(file_bytes, body.fileType, body.fileName)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    result["processing_time_seconds"] = round(time.perf_counter() - t0, 3)
    return result
