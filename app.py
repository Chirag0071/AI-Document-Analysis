"""
app.py — DocSense AI | Advanced Streamlit Frontend
HCL GUVI BuildBridge Hackathon 2026 — Track 2

Run: streamlit run app.py
"""

import streamlit as st
import requests
import base64
import json
import time
import os
from pathlib import Path

# ── Page config (MUST be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="DocSense AI — Document Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "sk_track2_987654321")
MAX_MB  = 150

EXT_MAP = {
    ".pdf":"pdf", ".docx":"docx", ".doc":"docx",
    ".jpg":"image", ".jpeg":"image", ".png":"image",
    ".bmp":"image", ".tiff":"image", ".tif":"image", ".webp":"image",
}

TECH_STACK = [
    "Groq Llama 3.3 70B", "spaCy en_core_web_lg", "FinBERT Sentiment",
    "Random Forest", "VADER + TextBlob", "YAKE Keyphrases",
    "Tesseract OCR", "Decision Tree Ensemble", "TF-IDF TextRank",
]

STEPS = [
    ("📥", "Reading file"),
    ("🔍", "Extracting text"),
    ("🧹", "Preprocessing"),
    ("🤖", "Groq AI analysis"),
    ("🏷️", "spaCy NER"),
    ("😊", "Sentiment ensemble"),
    ("📊", "Computing stats"),
]

EC = {
    "names":         ("Names",         "#67e8f9", "rgba(6,182,212,0.15)"),
    "organizations": ("Organizations", "#f9a8d4", "rgba(255,77,202,0.12)"),
    "dates":         ("Dates",         "#80ffe3", "rgba(0,255,194,0.1)"),
    "locations":     ("Locations",     "#fde68a", "rgba(251,191,36,0.1)"),
    "amounts":       ("Amounts",       "#fda4af", "rgba(255,61,90,0.12)"),
    "percentages":   ("Percentages",   "#fed7aa", "rgba(255,140,66,0.12)"),
    "emails":        ("Emails",        "#d4a0ff", "rgba(184,71,255,0.12)"),
    "phones":        ("Phones",        "#bbf7d0", "rgba(100,200,80,0.1)"),
    "urls":          ("URLs",          "#c7d2fe", "rgba(99,102,241,0.12)"),
}

# ── CSS (Streamlit-compatible — no animations, no JS) ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400;1,600&display=swap');

/* === ROOT THEME === */
html, body, [class*="css"], .stApp {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    background-color: #000000 !important;
    color: #d4ceff !important;
}
.stApp { background: #000000 !important; }
.block-container {
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 1180px !important;
    background: transparent !important;
}

/* === HIDE STREAMLIT DEFAULTS === */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* === HERO === */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    border-bottom: 1px solid rgba(184,71,255,0.15);
    margin-bottom: 2rem;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(184,71,255,0.06) 0%, transparent 70%);
    border-radius: 0 0 24px 24px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 20px;
    border: 1px solid rgba(184,71,255,0.3);
    border-radius: 30px;
    background: rgba(184,71,255,0.08);
    font-size: 11px;
    color: #b847ff;
    font-weight: 600;
    letter-spacing: 0.09em;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Lora', serif !important;
    font-size: clamp(2.2rem, 4vw, 3.6rem);
    color: #ffffff;
    line-height: 1.08;
    letter-spacing: -0.025em;
    margin-bottom: 14px;
}
.hero-em {
    font-style: italic;
    background: linear-gradient(120deg, #b847ff, #ff4dca, #b847ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    color: #4a4570;
    font-size: 0.97rem;
    max-width: 540px;
    margin: 0 auto 28px;
    line-height: 1.75;
    font-weight: 300;
}
.tech-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: center;
}
.tech-pill {
    padding: 5px 13px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.02);
    color: rgba(255,255,255,0.38);
}

/* === SECTION DIVIDER === */
.sec-div {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(184,71,255,0.3), transparent);
    margin: 1.5rem 0;
}

/* === UPLOAD SECTION === */
.upload-section {
    background: #04050e;
    border: 1px solid rgba(184,71,255,0.14);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.upload-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #b847ff 40%, #ff4dca 60%, transparent);
}
.upload-title {
    font-family: 'Lora', serif !important;
    font-size: 1.4rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 6px;
}
.upload-desc {
    font-size: 12px;
    color: #4a4570;
    margin-bottom: 16px;
    line-height: 1.6;
}
.fmt-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 14px;
}
.fmt-chip {
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.fmt-r { background: rgba(255,61,90,0.1); color: #ff8fa0; border: 1px solid rgba(255,61,90,0.2); }
.fmt-b { background: rgba(184,71,255,0.1); color: #d4a0ff; border: 1px solid rgba(184,71,255,0.2); }
.fmt-g { background: rgba(0,255,194,0.07); color: #80ffe3; border: 1px solid rgba(0,255,194,0.15); }

/* === FILE UPLOADED BAR === */
.file-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 13px 18px;
    background: rgba(184,71,255,0.07);
    border: 1px solid rgba(184,71,255,0.2);
    border-radius: 12px;
    margin: 10px 0;
}
.file-bar-icon { font-size: 24px; flex-shrink: 0; }
.file-bar-name { font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 2px; }
.file-bar-meta { font-size: 11px; color: #4a4570; }
.type-chip {
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    flex-shrink: 0;
    margin-left: auto;
}
.tc-pdf  { background: rgba(255,61,90,0.14);  color: #ff8fa0; border: 1px solid rgba(255,61,90,0.28); }
.tc-docx { background: rgba(184,71,255,0.14); color: #d4a0ff; border: 1px solid rgba(184,71,255,0.28); }
.tc-image{ background: rgba(0,255,194,0.08);  color: #80ffe3; border: 1px solid rgba(0,255,194,0.18); }

/* === CONFIG INPUTS === */
[data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 9px !important;
    color: #fff !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: rgba(184,71,255,0.5) !important;
    box-shadow: 0 0 0 1px rgba(184,71,255,0.2) !important;
}
[data-testid="stTextInput"] label {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #4a4570 !important;
}

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {
    background: rgba(7,9,26,0.95) !important;
    border: 1px solid rgba(184,71,255,0.22) !important;
    border-radius: 14px !important;
    padding: 0.8rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(184,71,255,0.5) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #4a4570 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* === ANALYZE BUTTON === */
.stButton > button {
    background: linear-gradient(135deg, #b847ff, #ff4dca) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s !important;
    position: relative !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:disabled { opacity: 0.22 !important; cursor: not-allowed !important; }

/* === PROGRESS === */
.stProgress > div > div { background: linear-gradient(90deg, #b847ff, #ff4dca) !important; border-radius: 3px !important; }
.stProgress > div { background: rgba(255,255,255,0.05) !important; border-radius: 3px !important; }

/* === RESULTS HEADER === */
.res-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 18px;
    border-bottom: 1px solid rgba(184,71,255,0.14);
    margin-bottom: 22px;
}
.res-title {
    font-family: 'Lora', serif !important;
    font-size: 2rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: -0.02em;
}
.res-ok {
    padding: 5px 15px;
    background: rgba(0,255,194,0.09);
    border: 1px solid rgba(0,255,194,0.24);
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    color: #00ffc2;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* === METRIC CARDS === */
.metric-card {
    background: #07091a;
    border: 1px solid rgba(184,71,255,0.13);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
    cursor: default;
    height: 100%;
}
.metric-card:hover { border-color: rgba(184,71,255,0.35); transform: translateY(-3px); }
.metric-top {
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #b847ff, #ff4dca);
}
.metric-num {
    font-family: 'Lora', serif !important;
    font-size: 2.2rem;
    font-weight: 600;
    color: #fff;
    line-height: 1;
    margin-bottom: 7px;
}
.metric-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a4570;
}

/* === CONTENT CARDS === */
.ds-card {
    background: #07091a;
    border: 1px solid rgba(184,71,255,0.12);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 12px;
    height: 100%;
    transition: border-color 0.2s;
}
.ds-card:hover { border-color: rgba(184,71,255,0.25); }
.ds-card-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(255,255,255,0.38);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ds-card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(184,71,255,0.12);
}
.sum-text {
    font-size: 14.5px;
    color: rgba(255,255,255,0.78);
    line-height: 1.85;
    font-weight: 300;
}
.dtype-badge {
    display: inline-block;
    padding: 4px 13px;
    background: rgba(232,200,74,0.08);
    border: 1px solid rgba(232,200,74,0.2);
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    color: #e8c84a;
    margin-top: 12px;
}

/* === SENTIMENT === */
.sent-label {
    font-family: 'Lora', serif !important;
    font-size: 2.3rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 18px;
}
.s-pos { color: #00ffc2; }
.s-neg { color: #ff3d5a; }
.s-neu { color: #ff8c42; }
.sbar-label {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #4a4570;
    margin-bottom: 4px;
}
.sbar-track {
    height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 10px;
}
.sbar-fill { height: 100%; border-radius: 3px; }

/* === PROFILE CARD === */
.profile-card {
    background: linear-gradient(135deg, rgba(184,71,255,0.08), rgba(255,77,202,0.05));
    border: 1px solid rgba(184,71,255,0.22);
    border-radius: 16px;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
}
.profile-avatar {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #b847ff, #ff4dca);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Lora', serif;
    font-size: 1.4rem; font-weight: 600; color: #fff;
    flex-shrink: 0;
}
.profile-name { font-family: 'Lora', serif !important; font-size: 1.25rem; font-weight: 600; color: #fff; margin-bottom: 3px; }
.profile-role { font-size: 12px; color: #4a4570; margin-bottom: 8px; }
.pcon {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    font-size: 11px;
    color: rgba(255,255,255,0.55);
    margin: 2px;
}

/* === SKILL TAGS === */
.skill-wrap { display: flex; flex-wrap: wrap; gap: 6px; }
.skill-tag {
    padding: 5px 13px;
    background: rgba(184,71,255,0.09);
    border: 1px solid rgba(184,71,255,0.22);
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: #d4a0ff;
}

/* === SECTION ITEMS === */
.sec-item {
    padding: 11px 0;
    border-bottom: 1px solid rgba(184,71,255,0.08);
    font-size: 13px;
    color: rgba(255,255,255,0.6);
    line-height: 1.6;
}
.sec-item:last-child { border-bottom: none; }

/* === ENTITY TAGS === */
.ent-group-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a4570;
    margin-bottom: 6px;
}
.etag {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
}

/* === KEY PHRASES === */
.kp-wrap { display: flex; flex-wrap: wrap; gap: 7px; }
.kpill {
    display: inline-block;
    padding: 6px 14px;
    background: rgba(184,71,255,0.08);
    border: 1px solid rgba(184,71,255,0.2);
    border-radius: 8px;
    font-size: 12px;
    color: #d4a0ff;
}

/* === READABILITY GRID === */
.read-item {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 13px;
    text-align: center;
    margin-bottom: 8px;
}
.read-val {
    font-family: 'Lora', serif !important;
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}
.read-key {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a4570;
    font-weight: 700;
}

/* === MISSING FIELDS === */
.missing-wrap {
    background: rgba(232,200,74,0.05);
    border: 1px solid rgba(232,200,74,0.2);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.missing-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #e8c84a;
    margin-bottom: 10px;
}
.miss-tag {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(232,200,74,0.08);
    border: 1px solid rgba(232,200,74,0.18);
    border-radius: 20px;
    font-size: 11px;
    color: #e8c84a;
    font-weight: 500;
    margin: 3px;
}

/* === STEP INDICATORS === */
.step-done { color: #00ffc2; font-weight: 600; }
.step-active { color: #b847ff; font-weight: 600; }
.step-pending { color: #4a4570; }

/* === EXPANDER === */
[data-testid="stExpander"] {
    background: #07091a !important;
    border: 1px solid rgba(184,71,255,0.12) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #4a4570 !important;
    font-size: 13px !important;
}

/* === SELECT/RADIO === */
[data-testid="stSelectbox"] select,
[data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.09) !important;
    color: #fff !important;
}

/* === STREAMLIT COLUMNS FIX === */
[data-testid="column"] { padding: 0 6px !important; }

/* === DOWNLOAD BUTTON === */
[data-testid="stDownloadButton"] > button {
    background: rgba(0,255,194,0.08) !important;
    border: 1px solid rgba(0,255,194,0.2) !important;
    color: #00ffc2 !important;
    border-radius: 10px !important;
    font-family: 'Bricolage Grotesque', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,255,194,0.15) !important;
}

/* === ERROR BOX === */
.err-box {
    background: rgba(255,61,90,0.07);
    border: 1px solid rgba(255,61,90,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    color: #ff8fa0;
    font-size: 13px;
    line-height: 1.6;
    margin: 12px 0;
    white-space: pre-wrap;
}

/* === SUCCESS BOX === */
.success-box {
    background: rgba(0,255,194,0.06);
    border: 1px solid rgba(0,255,194,0.2);
    border-radius: 12px;
    padding: 12px 18px;
    color: #00ffc2;
    font-size: 13px;
    margin: 8px 0;
}

/* === FOOTER === */
.app-footer {
    text-align: center;
    padding: 24px 0 8px;
    border-top: 1px solid rgba(184,71,255,0.12);
    margin-top: 40px;
    font-size: 12px;
    color: #4a4570;
}
.app-footer a { color: rgba(184,71,255,0.7); text-decoration: none; }
.app-footer a:hover { color: #b847ff; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def detect_type(name: str) -> str:
    ext = "." + name.split(".")[-1].lower()
    return EXT_MAP.get(ext, "unknown")


def card(title: str, icon: str, content: str):
    return f"""
    <div class="ds-card">
        <div class="ds-card-title">{icon} {title}</div>
        {content}
    </div>"""


def metric_card(num, label):
    return f"""
    <div class="metric-card">
        <div class="metric-top"></div>
        <div class="metric-num">{num}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def is_resume(r: dict) -> bool:
    dt = (r.get("document_type","")).lower()
    sm = (r.get("summary","")).lower()
    kp = " ".join(r.get("key_phrases",[])).lower()
    return any(w in dt+sm+kp for w in ["resume","cv","curriculum","intern","skills","education","experience"])


def extract_resume(r: dict, text: str) -> dict:
    t = text.lower()
    ents = r.get("entities", {})

    TECH = ["python","machine learning","deep learning","sql","fastapi","opencv","llm","c++","c/c++",
            "data visualization","data manipulation","javascript","react","node.js","tensorflow",
            "pytorch","keras","scikit-learn","pandas","numpy","docker","git","linux","azure","aws",
            "mongodb","mysql","nlp","computer vision","streamlit","flask","django","spark","hadoop"]
    skills = set()
    for k in TECH:
        if k in t:
            skills.add(k.replace("c/c++","C/C++").upper() if len(k)<=4 else k.title())
    for kp in r.get("key_phrases",[]):
        if 2 < len(kp) < 25:
            skills.add(kp.title())

    sents = [s.strip() for s in text.replace("\n"," ").split(".") if len(s.strip()) > 15]

    projects, experience, education, achievements = [], [], [], []
    for s in sents:
        sl = s.lower()
        if any(w in sl for w in ["project","built","developed","farmora","honeypot","agentic","created"]) and len(s)<300:
            projects.append(s)
        if any(w in sl for w in ["intern","engineer","developer","analyst","worked"]) and len(s)<250:
            experience.append(s)
        if any(w in sl for w in ["b.tech","bachelor","master","university","college","degree"]) and len(s)<300:
            education.append(s)
        if any(w in sl for w in ["finalist","winner","award","achievement","national","hackathon"]) and len(s)<300:
            achievements.append(s)

    urls = ents.get("urls",[])
    linkedin  = next((u for u in urls if "linkedin" in u.lower()),"")
    github    = next((u for u in urls if "github" in u.lower()),"")
    portfolio = next((u for u in urls if "linkedin" not in u.lower() and "github" not in u.lower()),"")

    missing = []
    if not linkedin:   missing.append("LinkedIn URL")
    if not github:     missing.append("GitHub URL")
    if not portfolio:  missing.append("Portfolio / Website")
    if not ents.get("emails"):  missing.append("Email address")
    if not ents.get("phones"):  missing.append("Phone number")
    if not any(w in t for w in ["intern","experience","worked"]): missing.append("Work / Internship experience")
    if "certif" not in t:       missing.append("Certifications")
    if not any(w in t for w in ["gpa","cgpa"]):  missing.append("GPA / CGPA")
    if "volunteer" not in t:    missing.append("Volunteer work")
    if not any(w in t for w in ["language","hindi","english","french","german"]): missing.append("Languages known")
    if "open source" not in t and "github.com/" not in t: missing.append("Open source contributions")

    return {
        "name": (ents.get("names",[""])[0]),
        "skills": list(skills),
        "emails": ents.get("emails",[]),
        "phones": ents.get("phones",[]),
        "locations": ents.get("locations",[]),
        "linkedin": linkedin, "github": github, "portfolio": portfolio,
        "projects": projects[:4],
        "experience": experience[:3],
        "education": education[:3],
        "achievements": achievements[:4],
        "missing": missing,
    }


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-badge">⬡ HCL GUVI BuildBridge Hackathon 2026 — Track 2</div>
  <div class="hero-title">Transform any document into <span class="hero-em">intelligence</span></div>
  <p class="hero-desc">Upload PDF, DOCX, or image up to {MAX_MB} MB. AI pipeline extracts summaries, entities, sentiment, keyphrases, and for resumes — skills, projects, experience and missing fields.</p>
  <div class="tech-wrap">
    {"".join(f'<span class="tech-pill">{t}</span>' for t in TECH_STACK)}
  </div>
</div>
""", unsafe_allow_html=True)


# ── UPLOAD SECTION ────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-section">
  <div class="upload-title">Upload your document</div>
  <p class="upload-desc">Supports PDF, DOCX, and image formats. Resumes get full profile extraction — skills, projects, experience, education, achievements and missing field detection.</p>
  <div class="fmt-row">
    <span class="fmt-chip fmt-r">PDF</span>
    <span class="fmt-chip fmt-b">DOCX</span>
    <span class="fmt-chip fmt-b">DOC</span>
    <span class="fmt-chip fmt-g">JPG</span>
    <span class="fmt-chip fmt-g">PNG</span>
    <span class="fmt-chip fmt-g">BMP</span>
    <span class="fmt-chip fmt-g">TIFF</span>
    <span class="fmt-chip fmt-g">WEBP</span>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    f"Drop file here or click to browse — max {MAX_MB} MB",
    type=["pdf","docx","doc","jpg","jpeg","png","bmp","tiff","webp"],
    label_visibility="visible",
)

# Config row
col_url, col_key = st.columns(2)
with col_url:
    api_url = st.text_input("API URL", value=API_URL)
with col_key:
    api_key = st.text_input("API Key", value=API_KEY, type="password")

st.markdown('<div style="font-size:11px;color:#4a4570;margin:-8px 0 12px">⚠ Keep defaults for local server. Change only for deployed API.</div>', unsafe_allow_html=True)

# ── File info ─────────────────────────────────────────────────────────────────
can_analyze = False

if uploaded:
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    ftype   = detect_type(uploaded.name)

    if size_mb > MAX_MB:
        st.markdown(f'<div class="err-box">⚠ File too large: {size_mb:.1f} MB. Maximum is {MAX_MB} MB.</div>', unsafe_allow_html=True)
    elif ftype == "unknown":
        st.markdown('<div class="err-box">Unsupported file format. Please upload PDF, DOCX, or an image.</div>', unsafe_allow_html=True)
    else:
        icons = {"pdf":"📕","docx":"📘","image":"🖼️"}
        chip_cls = {"pdf":"tc-pdf","docx":"tc-docx","image":"tc-image"}
        st.markdown(f"""
        <div class="file-bar">
          <div class="file-bar-icon">{icons.get(ftype,'📄')}</div>
          <div style="flex:1">
            <div class="file-bar-name">{uploaded.name}</div>
            <div class="file-bar-meta">{size_mb:.1f} MB &nbsp;•&nbsp; {ftype.upper()} detected &nbsp;•&nbsp; Ready to analyze</div>
          </div>
          <div class="type-chip {chip_cls.get(ftype,'')}">{ftype.upper()}</div>
        </div>""", unsafe_allow_html=True)
        can_analyze = True

# ── Analyze button ────────────────────────────────────────────────────────────
analyze_clicked = st.button(
    "⚡ Analyze Document",
    disabled=not can_analyze,
    use_container_width=True,
)

# ── Analysis pipeline ─────────────────────────────────────────────────────────
if analyze_clicked and can_analyze:
    ftype = detect_type(uploaded.name)

    # Progress bar + step indicators
    prog_bar  = st.progress(0)
    step_cols = st.columns(len(STEPS))

    def update_steps(active_idx):
        for i, (icon, label) in enumerate(STEPS):
            with step_cols[i]:
                if i < active_idx:
                    st.markdown(f'<div style="text-align:center;font-size:11px" class="step-done">✓ {label}</div>', unsafe_allow_html=True)
                elif i == active_idx:
                    st.markdown(f'<div style="text-align:center;font-size:11px" class="step-active">{icon} {label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align:center;font-size:11px" class="step-pending">○ {label}</div>', unsafe_allow_html=True)

    status_txt = st.empty()

    for i, (icon, label) in enumerate(STEPS):
        update_steps(i)
        prog_bar.progress(int((i + 1) / len(STEPS) * 82))
        status_txt.markdown(f'<div style="font-size:12px;color:#4a4570;text-align:center">{icon} {label}...</div>', unsafe_allow_html=True)
        time.sleep(0.28)

    try:
        # Encode file
        b64 = base64.b64encode(uploaded.getvalue()).decode()

        # Call API
        resp = requests.post(
            f"{api_url}/api/document-analyze",
            json={"fileName": uploaded.name, "fileType": ftype, "fileBase64": b64},
            headers={"Content-Type":"application/json","x-api-key":api_key},
            timeout=300,
        )
        resp.raise_for_status()
        r = resp.json()

        # Complete
        prog_bar.progress(100)
        update_steps(len(STEPS))
        status_txt.markdown('<div class="success-box">✓ Analysis complete!</div>', unsafe_allow_html=True)
        time.sleep(0.5)
        prog_bar.empty()
        status_txt.empty()
        for col in step_cols:
            col.empty()

        # ── SESSION STATE ─────────────────────────────────────────────────────
        st.session_state["result"]   = r
        st.session_state["filename"] = uploaded.name

    except requests.exceptions.ConnectionError:
        prog_bar.empty(); status_txt.empty()
        st.markdown('<div class="err-box">❌ Cannot connect to API server.\n\nMake sure it is running:\nuvicorn main:app --reload</div>', unsafe_allow_html=True)
    except Exception as e:
        prog_bar.empty(); status_txt.empty()
        st.markdown(f'<div class="err-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)


# ── RENDER RESULTS ────────────────────────────────────────────────────────────
if "result" in st.session_state:
    r  = st.session_state["result"]
    fn = st.session_state.get("filename","document")

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="res-header">
      <div class="res-title">Analysis Results</div>
      <div class="res-ok">✓ Complete</div>
    </div>""", unsafe_allow_html=True)

    # ── METRICS ───────────────────────────────────────────────────────────────
    st_obj = r.get("document_stats", {})
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, st_obj.get("word_count","-"), "Words"),
        (m2, st_obj.get("sentence_count","-"), "Sentences"),
        (m3, str(st_obj.get("reading_time_minutes","-"))+" min", "Min read"),
        (m4, str(r.get("processing_time_seconds","-"))+"s", "Process time"),
    ]:
        col.markdown(metric_card(val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RESUME PROFILE ────────────────────────────────────────────────────────
    raw_text = " ".join([
        r.get("summary",""),
        " ".join(r.get("key_phrases",[])),
        " ".join(str(v) for vals in r.get("entities",{}).values() for v in vals)
    ])

    if is_resume(r):
        rd2 = extract_resume(r, raw_text)
        initials = "".join(w[0] for w in rd2["name"].split()[:2]).upper() if rd2["name"] else "?"
        contacts = ""
        if rd2["emails"]:   contacts += f'<span class="pcon">✉ {rd2["emails"][0]}</span>'
        if rd2["phones"]:   contacts += f'<span class="pcon">☎ {rd2["phones"][0]}</span>'
        if rd2["locations"]:contacts += f'<span class="pcon">📍 {rd2["locations"][0]}</span>'
        if rd2["linkedin"]: contacts += f'<span class="pcon" style="color:#d4a0ff">🔗 LinkedIn</span>'
        if rd2["github"]:   contacts += f'<span class="pcon" style="color:#80ffe3">⌥ GitHub</span>'

        st.markdown(f"""
        <div class="profile-card">
          <div class="profile-avatar">{initials}</div>
          <div>
            <div class="profile-name">{rd2["name"] or "Name not detected"}</div>
            <div class="profile-role">{r.get("document_type","Resume / CV")}</div>
            <div>{contacts or '<span style="color:#4a4570">No contact info detected</span>'}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── SUMMARY + SENTIMENT ───────────────────────────────────────────────────
    cs, csent = st.columns([3,2])

    with cs:
        st.markdown(card("📝 AI Summary", "", f"""
        <div class="sum-text">{r.get("summary","No summary available.")}</div>
        <div class="dtype-badge">{r.get("document_type","Unknown")}</div>"""), unsafe_allow_html=True)

    with csent:
        sent = r.get("sentiment","Neutral")
        sc2  = r.get("sentiment_scores",{})
        pos  = round((sc2.get("Positive",0))*100)
        neu  = round((sc2.get("Neutral",0))*100)
        neg  = round((sc2.get("Negative",0))*100)
        scls = "s-pos" if sent=="Positive" else "s-neg" if sent=="Negative" else "s-neu"
        st.markdown(card("😊 Sentiment", "", f"""
        <div class="sent-label {scls}">{sent}</div>
        <div class="sbar-label"><span>Positive</span><span>{pos}%</span></div>
        <div class="sbar-track"><div class="sbar-fill" style="width:{pos}%;background:#00ffc2"></div></div>
        <div class="sbar-label"><span>Neutral</span><span>{neu}%</span></div>
        <div class="sbar-track"><div class="sbar-fill" style="width:{neu}%;background:#ff8c42"></div></div>
        <div class="sbar-label"><span>Negative</span><span>{neg}%</span></div>
        <div class="sbar-track"><div class="sbar-fill" style="width:{neg}%;background:#ff3d5a"></div></div>
        """), unsafe_allow_html=True)

    # ── RESUME SECTIONS ───────────────────────────────────────────────────────
    if is_resume(r):
        rd2 = extract_resume(r, raw_text)

        # Skills
        skills_html = "".join(f'<span class="skill-tag">{s}</span>' for s in rd2["skills"]) if rd2["skills"] else '<span style="color:#4a4570;font-size:13px">No specific skills detected.</span>'
        st.markdown(card("⚡ Skills Detected", "", f'<div class="skill-wrap">{skills_html}</div>'), unsafe_allow_html=True)

        # Projects + Experience
        cp, ce = st.columns(2)
        with cp:
            proj_html = "".join(f'<div class="sec-item">{p}</div>' for p in rd2["projects"]) if rd2["projects"] else '<div style="color:#4a4570;font-size:13px;padding:8px 0">No projects detected.</div>'
            st.markdown(card("💻 Projects", "", proj_html), unsafe_allow_html=True)
        with ce:
            exp_html = "".join(f'<div class="sec-item">{e}</div>' for e in rd2["experience"]) if rd2["experience"] else '<div style="color:#4a4570;font-size:13px;padding:8px 0">No experience detected.</div>'
            st.markdown(card("💼 Experience", "", exp_html), unsafe_allow_html=True)

        # Education + Achievements
        ced, cac = st.columns(2)
        with ced:
            edu_html = "".join(f'<div class="sec-item">{e}</div>' for e in rd2["education"]) if rd2["education"] else '<div style="color:#4a4570;font-size:13px;padding:8px 0">No education info detected.</div>'
            st.markdown(card("🎓 Education", "", edu_html), unsafe_allow_html=True)
        with cac:
            ach_html = "".join(f'<div class="sec-item">{a}</div>' for a in rd2["achievements"]) if rd2["achievements"] else '<div style="color:#4a4570;font-size:13px;padding:8px 0">No achievements detected.</div>'
            st.markdown(card("🏆 Achievements", "", ach_html), unsafe_allow_html=True)

        # Missing fields
        if rd2["missing"]:
            tags = "".join(f'<span class="miss-tag">{m}</span>' for m in rd2["missing"])
            st.markdown(f"""
            <div class="missing-wrap">
              <div class="missing-title">⚠ Missing fields — add these to strengthen your resume</div>
              <div>{tags}</div>
            </div>""", unsafe_allow_html=True)

    # ── ENTITIES ──────────────────────────────────────────────────────────────
    ents = r.get("entities",{})
    ent_html = ""
    has_ents = False
    for k,(lbl,fg,bg) in EC.items():
        vals = ents.get(k,[])
        if not vals: continue
        has_ents = True
        tags = "".join(f'<span class="etag" style="background:{bg};color:{fg}">{v}</span>' for v in vals)
        ent_html += f'<div style="margin-bottom:12px"><div class="ent-group-title">{lbl}</div><div>{tags}</div></div>'
    if not has_ents:
        ent_html = '<span style="color:#4a4570;font-size:13px">No entities detected.</span>'
    st.markdown(card("🏷️ Named Entities", "", f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px">{ent_html}</div>'), unsafe_allow_html=True)

    # ── KEYPHRASES + READABILITY ──────────────────────────────────────────────
    ck, cr = st.columns(2)

    with ck:
        kp = r.get("key_phrases",[])
        pills = "".join(f'<span class="kpill">{k}</span>' for k in kp) if kp else '<span style="color:#4a4570;font-size:13px">None detected.</span>'
        st.markdown(card("🔑 Key Phrases","", f'<div class="kp-wrap">{pills}</div>'), unsafe_allow_html=True)

    with cr:
        rd3 = st_obj.get("readability",{})
        read_items = [
            (rd3.get("flesch_reading_ease","-"), "#b847ff", "Flesch Ease"),
            (rd3.get("flesch_kincaid_grade","-"), "#ff4dca", "FK Grade"),
            (rd3.get("gunning_fog_index","-"), "#ff3d5a", "Gunning Fog"),
            (rd3.get("interpretation","-"), "#00ffc2", "Interpretation"),
        ]
        ri_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
        for val, clr, lbl in read_items:
            ri_html += f'<div class="read-item"><div class="read-val" style="color:{clr}">{val}</div><div class="read-key">{lbl}</div></div>'
        ri_html += f'</div><div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(184,71,255,0.1);font-size:11px;color:#4a4570;display:flex;gap:18px"><span>Language: <b style="color:rgba(255,255,255,0.65)">{st_obj.get("language","en").upper()}</b></span><span>Lex div: <b style="color:rgba(255,255,255,0.65)">{st_obj.get("lexical_diversity",0):.3f}</b></span></div>'
        st.markdown(card("📈 Readability","", ri_html), unsafe_allow_html=True)

    # ── RAW JSON ──────────────────────────────────────────────────────────────
    with st.expander("View raw JSON response"):
        st.json(r)

    # ── DOWNLOAD ──────────────────────────────────────────────────────────────
    json_str = json.dumps(r, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download full JSON result",
        data=json_str,
        file_name=Path(fn).stem + "_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

elif not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem">
      <div style="font-size:4.5rem;margin-bottom:1rem;opacity:0.4">⬡</div>
      <div style="font-size:15px;color:#4a4570;font-weight:300">Upload a document above to begin analysis</div>
      <div style="font-size:12px;color:#2a2550;margin-top:8px">PDF &bull; DOCX &bull; JPG &bull; PNG &bull; BMP &bull; TIFF &bull; WEBP</div>
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  HCL GUVI BuildBridge Hackathon 2026 &mdash; Track 2: AI Document Analysis &nbsp;&bull;&nbsp;
  <a href="https://github.com/Chirag0071/AI-Document-Analysis" target="_blank">GitHub</a>
  &nbsp;&bull;&nbsp;
  <a href="http://127.0.0.1:8000/docs" target="_blank">API Docs</a>
</div>""", unsafe_allow_html=True)