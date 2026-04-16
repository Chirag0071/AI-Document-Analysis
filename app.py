"""
DocSense AI  ·  Streamlit Frontend
HCL GUVI BuildBridge Hackathon 2026  ·  Track 2

Run frontend : streamlit run app.py
Run backend  : uvicorn main:app --reload
"""

import streamlit as st
import requests
import base64
import json
import time
import os
from pathlib import Path

st.set_page_config(
    page_title="DocSense AI · Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "sk_track2_987654321")
MAX_MB  = 150

EXT_MAP = {
    ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".bmp": "image", ".tiff": "image", ".tif": "image", ".webp": "image",
}

PIPELINE_STEPS = [
    ("01","Reading"), ("02","Extracting"), ("03","Cleaning"),
    ("04","Groq AI"), ("05","Entities"),   ("06","Sentiment"), ("07","Finalising"),
]

ENTITY_MAP = {
    "names":         ("People",        "#1a1a1a", "#f5e6c8"),
    "organizations": ("Organizations", "#1a1a1a", "#ffd6a5"),
    "dates":         ("Dates",         "#1a1a1a", "#c8e6c9"),
    "locations":     ("Locations",     "#1a1a1a", "#bbdefb"),
    "amounts":       ("Amounts",       "#1a1a1a", "#f8bbd9"),
    "percentages":   ("Percentages",   "#1a1a1a", "#e1bee7"),
    "emails":        ("Emails",        "#1a1a1a", "#b2ebf2"),
    "phones":        ("Phones",        "#1a1a1a", "#dcedc8"),
    "urls":          ("URLs",          "#1a1a1a", "#ffe0b2"),
}

RESUME_TECH = [
    "python","machine learning","deep learning","sql","fastapi","opencv","llm",
    "javascript","react","tensorflow","pytorch","pandas","numpy","docker","git",
    "azure","aws","mongodb","mysql","nlp","computer vision","streamlit","flask",
    "django","c/c++","data visualization","node.js","scikit-learn","keras",
    "figma","photoshop","illustrator","web design","adobe",
]


# ── Safe helpers ──────────────────────────────────────────────────────────────
def safe_first(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return str(lst[0])
    return ""

def safe_list(v):
    return v if isinstance(v, list) else []

def safe_str(v, default=""):
    return str(v) if v is not None else default

def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

def safe_dict(v):
    return v if isinstance(v, dict) else {}

def detect_type(filename):
    if "." not in filename:
        return "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    return EXT_MAP.get(ext, "unknown")

def is_resume(r):
    combined = " ".join([
        safe_str(r.get("document_type")),
        safe_str(r.get("summary")),
        " ".join(safe_list(r.get("key_phrases"))),
    ]).lower()
    return any(w in combined for w in
               ["resume","cv","curriculum","intern","skills","education",
                "graphic designer","bachelor","university"])

def extract_resume(r, raw_text):
    t    = raw_text.lower()
    ents = safe_dict(r.get("entities"))
    skills = set()
    for k in RESUME_TECH:
        if k in t:
            skills.add(k.upper() if len(k) <= 4 else k.title())
    for kp in safe_list(r.get("key_phrases")):
        if isinstance(kp, str) and 2 < len(kp) < 30:
            skills.add(kp.strip())
    sents = [s.strip() for s in raw_text.replace("\n", " ").split(".")
             if isinstance(s, str) and len(s.strip()) > 20]
    def filt(kws, n):
        return [s for s in sents
                if any(k in s.lower() for k in kws) and 20 < len(s) < 340][:n]
    urls      = safe_list(ents.get("urls"))
    linkedin  = next((u for u in urls if "linkedin"  in str(u).lower()), "")
    github    = next((u for u in urls if "github"    in str(u).lower()), "")
    portfolio = next((u for u in urls
                      if "linkedin" not in str(u).lower()
                      and "github"  not in str(u).lower()), "")
    missing = []
    if not linkedin:                          missing.append("LinkedIn URL")
    if not github:                            missing.append("GitHub URL")
    if not portfolio:                         missing.append("Portfolio/Website")
    if not safe_list(ents.get("emails")):     missing.append("Email address")
    if not safe_list(ents.get("phones")):     missing.append("Phone number")
    if not any(w in t for w in ["intern","experience","worked","designer","engineer"]):
        missing.append("Work Experience")
    if "certif" not in t:                     missing.append("Certifications")
    if not any(w in t for w in ["gpa","cgpa"]):
        missing.append("GPA / CGPA")
    return {
        "name":         safe_first(ents.get("names")),
        "skills":       sorted(skills),
        "emails":       safe_list(ents.get("emails")),
        "phones":       safe_list(ents.get("phones")),
        "locations":    safe_list(ents.get("locations")),
        "linkedin":     linkedin,
        "github":       github,
        "portfolio":    portfolio,
        "projects":     filt(["project","built","developed","campaign","brand","portfolio","social media"],4),
        "experience":   filt(["intern","engineer","developer","analyst","designer","senior","worked"],3),
        "education":    filt(["b.tech","bachelor","master","university","college","degree","graduated"],3),
        "achievements": filt(["finalist","winner","award","achievement","hackathon","boosted","increased"],4),
        "missing":      missing,
    }


# ── Colour helpers ────────────────────────────────────────────────────────────
def am(v):
    return f"rgba(212,175,95,{v})"


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

html,body,[class*="css"],.stApp{{
    font-family:'Sora',sans-serif!important;
    background:#0e0c07!important;
    color:#e8dfc8!important;
}}
.stApp{{background:#0e0c07!important;}}
.block-container{{padding:0 2rem 6rem!important;max-width:1200px!important;margin:0 auto!important;}}
#MainMenu,footer,header,.stDeployButton,
[data-testid="stToolbar"],[data-testid="stDecoration"]{{display:none!important;}}

[data-testid="stFileUploader"]{{
    background:{am('.04')}!important;
    border:1.5px dashed {am('.35')}!important;
    border-radius:12px!important;
}}
[data-testid="stFileUploader"]:hover{{
    border-color:{am('.65')}!important;
    background:{am('.06')}!important;
}}
[data-testid="stFileUploaderDropzoneInstructions"]{{color:#6b5e3e!important;}}

[data-testid="stTextInput"]>div>div>input{{
    background:{am('.06')}!important;
    border:1px solid {am('.25')}!important;
    border-radius:8px!important;
    color:#e8dfc8!important;
    font-family:'IBM Plex Mono',monospace!important;
    font-size:12px!important;
}}
[data-testid="stTextInput"]>div>div>input:focus{{
    border-color:{am('.65')}!important;
    box-shadow:0 0 0 2px {am('.12')}!important;
}}
[data-testid="stTextInput"] label{{
    font-family:'IBM Plex Mono',monospace!important;
    font-size:9px!important;font-weight:500!important;
    text-transform:uppercase!important;letter-spacing:.18em!important;
    color:#6b5e3e!important;
}}

.stButton>button{{
    background:#d4af5f!important;
    color:#0e0c07!important;
    border:none!important;border-radius:8px!important;
    font-family:'Sora',sans-serif!important;font-weight:700!important;
    font-size:14px!important;padding:.8rem 2rem!important;width:100%!important;
    letter-spacing:.04em!important;transition:all .2s!important;
    box-shadow:0 2px 16px {am('.25')}!important;
}}
.stButton>button:hover:not(:disabled){{
    background:#e0c070!important;
    box-shadow:0 4px 24px {am('.4')}!important;
    transform:translateY(-1px)!important;
}}
.stButton>button:disabled{{
    background:{am('.15')}!important;
    color:{am('.3')}!important;
    box-shadow:none!important;
}}

.stProgress>div>div{{
    background:linear-gradient(90deg,#d4af5f,#f0cc70,#d4af5f)!important;
    background-size:200% 100%!important;
    border-radius:2px!important;
    animation:gold_shimmer 1.5s infinite!important;
}}
@keyframes gold_shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}
.stProgress>div{{
    background:{am('.1')}!important;height:3px!important;border-radius:2px!important;
}}

[data-testid="stExpander"]{{
    background:{am('.04')}!important;
    border:1px solid {am('.15')}!important;
    border-radius:10px!important;margin-top:8px!important;
}}
[data-testid="stExpander"] summary{{
    font-family:'IBM Plex Mono',monospace!important;
    font-size:11px!important;color:#8a7048!important;
}}

[data-testid="stDownloadButton"]>button{{
    background:{am('.1')}!important;
    border:1px solid {am('.3')}!important;
    color:#d4af5f!important;border-radius:8px!important;
    box-shadow:none!important;
    font-family:'IBM Plex Mono',monospace!important;
    font-size:12px!important;font-weight:500!important;
}}
[data-testid="stDownloadButton"]>button:hover{{
    background:{am('.18')}!important;
    border-color:{am('.5')}!important;
}}

[data-testid="column"]{{padding:0 6px!important;}}

@keyframes pulse_gold{{
    0%,100%{{box-shadow:0 0 0 0 {am('.3')}}}
    50%{{box-shadow:0 0 0 6px transparent}}
}}
</style>
""", unsafe_allow_html=True)


# ── HTML component builders ───────────────────────────────────────────────────
def ds_card(icon, title, body, accent="#d4af5f"):
    return f"""
<div style="background:#13100a;border:1px solid {am('.18')};border-radius:14px;
     padding:22px;height:100%;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,{accent},transparent);opacity:.55;"></div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:15px;
    padding-bottom:12px;border-bottom:1px solid {am('.08')};">
    <div style="background:{accent}22;width:32px;height:32px;border-radius:8px;
      display:flex;align-items:center;justify-content:center;font-size:15px;">{icon}</div>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:500;
      letter-spacing:.2em;text-transform:uppercase;color:{am('.45')};">{title}</span>
  </div>
  {body}
</div>"""

def metric_tile(value, label, accent="#d4af5f"):
    return f"""
<div style="background:#13100a;border:1px solid {am('.18')};border-radius:12px;
     padding:20px 14px;text-align:center;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:25%;right:25%;height:1px;
    background:linear-gradient(90deg,transparent,{accent},transparent);"></div>
  <div style="font-family:'Libre Baskerville',serif;font-size:2.1rem;
    color:{accent};line-height:1;margin-bottom:7px;">{value}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
    text-transform:uppercase;letter-spacing:.16em;color:{am('.4')};">{label}</div>
</div>"""

def amber_tag(text):
    return (f'<span style="display:inline-block;padding:4px 11px;border-radius:20px;'
            f'font-size:11px;font-weight:500;background:{am(".1")};'
            f'border:1px solid {am(".3")};color:#d4af5f;margin:2px;">{text}</span>')

def colored_tag(text, fg, bg):
    return (f'<span style="display:inline-block;padding:3px 9px;border-radius:6px;'
            f'font-size:12px;font-weight:500;background:{bg};color:{fg};margin:2px;">{text}</span>')

def miss_tag(text):
    return (f'<span style="display:inline-block;padding:4px 12px;border-radius:20px;'
            f'font-size:10px;background:rgba(255,160,50,.1);'
            f'border:1px solid rgba(255,160,50,.28);color:#ffaa40;margin:2px;">{text}</span>')

def kp_tag(text):
    return (f'<span style="display:inline-block;padding:5px 13px;border-radius:6px;'
            f'font-size:11px;background:rgba(168,213,162,.08);'
            f'border:1px solid rgba(168,213,162,.2);color:#a8d5a2;margin:3px;">{text}</span>')

def sec_row(text):
    return (f'<div style="padding:10px 0;border-bottom:1px solid {am(".07")};'
            f'font-size:12px;color:{am(".75")};line-height:1.7;">{text}</div>')

def rule():
    return f'<div style="height:1px;background:{am(".12")};margin:1.8rem 0;"></div>'

def section_hdr(text):
    return (f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.2rem;">'
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
            f'font-weight:500;letter-spacing:.22em;text-transform:uppercase;'
            f'color:{am(".4")};">{text}</span>'
            f'<span style="flex:1;height:1px;background:{am(".1")};"></span>'
            f'</div>')

def corner_box():
    c = am('.3')
    return (f'<div style="position:absolute;top:12px;left:12px;width:14px;height:14px;'
            f'border-top:1px solid {c};border-left:1px solid {c};"></div>'
            f'<div style="position:absolute;top:12px;right:12px;width:14px;height:14px;'
            f'border-top:1px solid {c};border-right:1px solid {c};"></div>'
            f'<div style="position:absolute;bottom:12px;left:12px;width:14px;height:14px;'
            f'border-bottom:1px solid {c};border-left:1px solid {c};"></div>'
            f'<div style="position:absolute;bottom:12px;right:12px;width:14px;height:14px;'
            f'border-bottom:1px solid {c};border-right:1px solid {c};"></div>')


# ── HEADER BAR ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
  padding:18px 0 20px;border-bottom:1px solid {am('.15')};margin-bottom:0;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:36px;height:36px;background:{am('.12')};border-radius:10px;
      border:1px solid {am('.3')};display:flex;align-items:center;
      justify-content:center;font-size:18px;">◈</div>
    <div>
      <div style="font-family:'Libre Baskerville',serif;font-size:1.25rem;
        color:#e8dfc8;letter-spacing:-.01em;">
        DocSense <em style="font-style:italic;color:#d4af5f;">AI</em></div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
        color:{am('.4')};letter-spacing:.12em;text-transform:uppercase;margin-top:1px;">
        Document Intelligence Platform</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:20px;">
    <a href="http://127.0.0.1:8000/docs" target="_blank"
       style="font-family:'IBM Plex Mono',monospace;font-size:10px;
       color:{am('.45')};text-decoration:none;letter-spacing:.06em;">API Docs ↗</a>
    <a href="https://github.com/Chirag0071/AI-Document-Analysis" target="_blank"
       style="font-family:'IBM Plex Mono',monospace;font-size:10px;
       color:{am('.45')};text-decoration:none;letter-spacing:.06em;">GitHub ↗</a>
  </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────────
tech_items = [
    "Groq Llama 3.3 70B","spaCy NER","FinBERT","VADER+TextBlob",
    "YAKE Keyphrases","Tesseract OCR","PyMuPDF","python-docx",
]
tech_html = " ".join(
    f'<span style="padding:3px 11px;border-radius:4px;font-family:\'IBM Plex Mono\',monospace;'
    f'font-size:9px;border:1px solid {am(".2")};color:{am(".55")};background:{am(".04")};">{t}</span>'
    for t in tech_items
)

st.markdown(f"""
<div style="padding:4rem 0 3rem;text-align:center;position:relative;">
  {corner_box()}
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
    letter-spacing:.2em;text-transform:uppercase;color:{am('.4')};margin-bottom:1.5rem;">
    ◈ HCL GUVI BuildBridge Hackathon 2026 — Track 2
  </div>
  <h1 style="font-family:'Libre Baskerville',serif;
    font-size:clamp(2.8rem,5.5vw,5rem);color:#f5ead8;
    line-height:1.04;letter-spacing:-.03em;margin:0 0 1.2rem;">
    Every document holds<br>
    <em style="font-style:italic;color:#d4af5f;">structured intelligence</em>
  </h1>
  <p style="font-size:14px;color:{am('.5')};max-width:500px;
    margin:0 auto 2.5rem;line-height:1.85;">
    Upload PDF · DOCX · Image — AI pipeline extracts<br>
    summaries · entities · sentiment · keyphrases
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;">
    {tech_html}
  </div>
</div>
""", unsafe_allow_html=True)


# ── API STATUS ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=25)
def ping(url):
    try:
        r = requests.get(url + "/health", timeout=3)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False

online = ping(API_URL)
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
  padding:10px 18px;background:{am('.05')};border:1px solid {am('.15')};
  border-radius:8px;margin-bottom:2rem;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
    letter-spacing:.1em;text-transform:uppercase;color:{am('.4')};">System Status</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
    color:{'#6fcf97' if online else '#eb5757'};">
    {'● API ONLINE' if online else '○ API OFFLINE — run: uvicorn main:app --reload'}
  </span>
</div>
""", unsafe_allow_html=True)


# ── UPLOAD PANEL ──────────────────────────────────────────────────────────────
fmt_items = [
    ("PDF",  "#fca5a5","rgba(239,68,68,.09)","rgba(239,68,68,.25)"),
    ("DOCX", "#d4af5f",am(".09"),am(".3")),
    ("DOC",  "#d4af5f",am(".09"),am(".3")),
    ("JPG",  "#a8d5a2","rgba(120,190,120,.09)","rgba(120,190,120,.3)"),
    ("PNG",  "#a8d5a2","rgba(120,190,120,.09)","rgba(120,190,120,.3)"),
    ("TIFF", "#a8d5a2","rgba(120,190,120,.09)","rgba(120,190,120,.3)"),
    ("WEBP", "#a8d5a2","rgba(120,190,120,.09)","rgba(120,190,120,.3)"),
]
fmt_html = " ".join(
    f'<span style="padding:3px 10px;border-radius:4px;font-family:\'IBM Plex Mono\',monospace;'
    f'font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;'
    f'background:{bg};color:{fg};border:1px solid {bd};">{label}</span>'
    for label, fg, bg, bd in fmt_items
)
st.markdown(f"""
<div style="background:#13100a;border:1px solid {am('.22')};border-radius:16px;
     padding:28px 28px 22px;position:relative;overflow:hidden;margin-bottom:14px;">
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,#d4af5f 40%,#f0cc70 60%,transparent);"></div>
  <div style="font-family:'Libre Baskerville',serif;font-size:1.75rem;
    color:#f5ead8;margin-bottom:8px;">Upload your document</div>
  <p style="font-family:'IBM Plex Mono',monospace;font-size:10px;
    color:{am('.45')};margin-bottom:18px;line-height:1.7;letter-spacing:.02em;">
    PDF · DOCX · DOC · JPG · PNG · BMP · TIFF · WEBP — max 150 MB<br>
    Resume files receive enhanced profile extraction automatically
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:5px;">{fmt_html}</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop file here or click to browse",
    type=["pdf","docx","doc","jpg","jpeg","png","bmp","tiff","webp"],
    label_visibility="visible",
)

col_u, col_k = st.columns(2)
with col_u:
    api_url = st.text_input("API URL", value=API_URL)
with col_k:
    api_key = st.text_input("API Key", value=API_KEY, type="password")

st.markdown(
    f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
    f'color:{am(".35")};margin:-6px 0 16px;letter-spacing:.05em;">⚠ Keep defaults for local server.</p>',
    unsafe_allow_html=True,
)

# ── File info bar ──────────────────────────────────────────────────────────────
can_analyze = False
if uploaded is not None:
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    ftype   = detect_type(uploaded.name)
    if size_mb > MAX_MB:
        st.markdown(
            f'<div style="background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.3);'
            f'border-radius:10px;padding:12px 18px;color:#fca5a5;font-size:12px;'
            f'font-family:\'IBM Plex Mono\',monospace;">⚠ File too large: {size_mb:.1f} MB. Max {MAX_MB} MB.</div>',
            unsafe_allow_html=True,
        )
    elif ftype == "unknown":
        st.markdown(
            '<div style="background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.3);'
            'border-radius:10px;padding:12px 18px;color:#fca5a5;font-size:12px;'
            'font-family:\'IBM Plex Mono\',monospace;">⚠ Unsupported format.</div>',
            unsafe_allow_html=True,
        )
    else:
        ico_map = {"pdf":"📕","docx":"📘","image":"🖼️"}
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;padding:13px 18px;
          background:{am('.06')};border:1px solid {am('.28')};border-radius:10px;margin-bottom:12px;">
          <span style="font-size:24px;">{ico_map.get(ftype,'📄')}</span>
          <div style="flex:1;">
            <div style="font-size:13px;font-weight:600;color:#f5ead8;margin-bottom:2px;">{uploaded.name}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{am('.45')};">
              {size_mb:.1f} MB &nbsp;·&nbsp; {ftype.upper()} &nbsp;·&nbsp; Ready
            </div>
          </div>
          <span style="font-family:'IBM Plex Mono',monospace;padding:3px 13px;border-radius:20px;
            font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
            background:{am('.12')};border:1px solid {am('.35')};color:#d4af5f;">{ftype.upper()}</span>
        </div>""", unsafe_allow_html=True)
        can_analyze = True


# ── Analyze button ─────────────────────────────────────────────────────────────
clicked = st.button("◈  Analyze Document", disabled=not can_analyze, use_container_width=True)


# ── Pipeline ───────────────────────────────────────────────────────────────────
if clicked and can_analyze and uploaded is not None:
    ftype = detect_type(uploaded.name)

    def pipeline_html(active):
        items = ""
        for i, (num, label) in enumerate(PIPELINE_STEPS):
            if i < active:
                ns = f"color:#6fcf97;border-color:rgba(111,207,151,.4);background:rgba(111,207,151,.08);"
                lc = "#6fcf97"; ns2 = num_show = "✓"
            elif i == active:
                ns = f"color:#d4af5f;border-color:{am('.45')};background:{am('.1')};animation:pulse_gold .8s ease-in-out infinite;"
                lc = "#d4af5f"; ns2 = num; num_show = num
            else:
                ns = f"color:{am('.2')};border-color:{am('.1')};background:transparent;"
                lc = am(".2"); ns2 = num; num_show = num
            items += (
                f'<div style="display:flex;flex-direction:column;align-items:center;gap:5px;flex:1;">'
                f'<div style="width:28px;height:28px;border-radius:50%;border:1px solid;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-family:\'IBM Plex Mono\',monospace;font-size:10px;font-weight:600;{ns}">'
                f'{"✓" if i < active else num}</div>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
                f'color:{lc};letter-spacing:.04em;">{label}</span>'
                f'</div>'
            )
        return (
            f'<div style="display:flex;align-items:flex-start;'
            f'background:{am(".04")};border:1px solid {am(".12")};'
            f'border-radius:10px;padding:16px 12px;margin:10px 0 18px;">'
            f'{items}</div>'
        )

    prog    = st.progress(0)
    pipe_ph = st.empty()
    stat_ph = st.empty()

    for i, (num, label) in enumerate(PIPELINE_STEPS):
        pipe_ph.markdown(pipeline_html(i), unsafe_allow_html=True)
        prog.progress(int((i + 1) / len(PIPELINE_STEPS) * 82))
        stat_ph.markdown(
            f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
            f'color:{am(".45")};text-align:center;letter-spacing:.06em;">'
            f'{num} / 07 — {label}…</p>',
            unsafe_allow_html=True,
        )
        time.sleep(0.22)

    try:
        b64  = base64.b64encode(uploaded.getvalue()).decode()
        resp = requests.post(
            f"{api_url}/api/document-analyze",
            json={"fileName": uploaded.name, "fileType": ftype, "fileBase64": b64},
            headers={"Content-Type": "application/json", "x-api-key": api_key},
            timeout=300,
        )
        resp.raise_for_status()
        result = resp.json()

        prog.progress(100)
        pipe_ph.markdown(pipeline_html(len(PIPELINE_STEPS)), unsafe_allow_html=True)
        stat_ph.markdown(
            f'<div style="background:rgba(111,207,151,.07);border:1px solid rgba(111,207,151,.25);'
            f'border-radius:8px;padding:10px 16px;font-family:\'IBM Plex Mono\',monospace;'
            f'color:#6fcf97;font-size:11px;text-align:center;letter-spacing:.06em;">◈ Analysis complete</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.4)
        prog.empty(); stat_ph.empty(); pipe_ph.empty()
        st.session_state["result"]   = result
        st.session_state["filename"] = uploaded.name

    except requests.exceptions.ConnectionError:
        prog.empty(); stat_ph.empty(); pipe_ph.empty()
        st.markdown(
            '<div style="background:rgba(235,87,87,.07);border:1px solid rgba(235,87,87,.3);'
            'border-radius:10px;padding:14px 18px;font-family:\'IBM Plex Mono\',monospace;'
            'color:#fca5a5;font-size:12px;line-height:1.7;">'
            '✗ Cannot connect to API.<br>Start backend: <strong>uvicorn main:app --reload</strong></div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        prog.empty(); stat_ph.empty(); pipe_ph.empty()
        st.markdown(
            f'<div style="background:rgba(235,87,87,.07);border:1px solid rgba(235,87,87,.3);'
            f'border-radius:10px;padding:14px 18px;font-family:\'IBM Plex Mono\',monospace;'
            f'color:#fca5a5;font-size:12px;">✗ {exc}</div>',
            unsafe_allow_html=True,
        )


# ── RENDER RESULTS ─────────────────────────────────────────────────────────────
if "result" in st.session_state:
    r  = st.session_state["result"]
    fn = st.session_state.get("filename", "document")

    if not isinstance(r, dict):
        st.error("Invalid API response.")
        st.stop()

    st.markdown(rule(), unsafe_allow_html=True)

    doc_type = safe_str(r.get("document_type"), "Unknown")

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;justify-content:space-between;
      margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid {am('.12')};">
      <div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
          letter-spacing:.2em;text-transform:uppercase;color:{am('.4')};margin-bottom:8px;">
          Analysis Report
        </div>
        <div style="font-family:'Libre Baskerville',serif;font-size:2.4rem;
          color:#f5ead8;letter-spacing:-.025em;line-height:1.05;">
          Document <em style="font-style:italic;color:#d4af5f;">Intelligence</em>
        </div>
      </div>
      <div style="text-align:right;display:flex;flex-direction:column;gap:6px;align-items:flex-end;">
        <span style="padding:5px 14px;background:rgba(111,207,151,.08);
          border:1px solid rgba(111,207,151,.25);border-radius:20px;
          font-family:'IBM Plex Mono',monospace;font-size:9px;
          color:#6fcf97;text-transform:uppercase;letter-spacing:.12em;">✓ Complete</span>
        <span style="padding:5px 14px;background:{am('.08')};border:1px solid {am('.28')};
          border-radius:20px;font-family:'IBM Plex Mono',monospace;font-size:9px;
          color:#d4af5f;text-transform:uppercase;letter-spacing:.12em;">{doc_type}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── metrics ───────────────────────────────────────────────────────────────
    stats = safe_dict(r.get("document_stats"))
    m1, m2, m3, m4 = st.columns(4)
    accs = ["#d4af5f","#f0cc70","#a8c8a0","#d4af5f"]
    for col, val, lbl, ac in [
        (m1, safe_str(stats.get("word_count"),"-"),      "Words",       accs[0]),
        (m2, safe_str(stats.get("sentence_count"),"-"),  "Sentences",   accs[1]),
        (m3, safe_str(stats.get("reading_time_minutes"),"-")+" min","Read Time",accs[2]),
        (m4, safe_str(r.get("processing_time_seconds"),"-")+"s","Process Time",accs[3]),
    ]:
        col.markdown(metric_tile(val, lbl, ac), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── raw text for resume ───────────────────────────────────────────────────
    ents_raw = safe_dict(r.get("entities"))
    kp_list  = safe_list(r.get("key_phrases"))
    raw_text = " ".join(filter(None, [
        safe_str(r.get("summary")),
        " ".join(safe_str(v) for v in kp_list),
        " ".join(safe_str(item) for vals in ents_raw.values() for item in safe_list(vals)),
    ]))

    # ── resume profile card ───────────────────────────────────────────────────
    resume_mode = is_resume(r)
    rd2 = {}

    if resume_mode:
        rd2      = extract_resume(r, raw_text)
        name_str = safe_str(rd2.get("name"), "Name not detected")
        initials = "".join(w[0] for w in name_str.split()[:2] if w).upper() or "?"

        contacts = ""
        em = safe_first(rd2.get("emails"))
        ph = safe_first(rd2.get("phones"))
        lc = safe_first(rd2.get("locations"))
        if em: contacts += f'<span style="padding:3px 10px;background:{am(".08")};border:1px solid {am(".25")};border-radius:6px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:{am(".7")};margin:2px;">✉ {em}</span>'
        if ph: contacts += f'<span style="padding:3px 10px;background:{am(".08")};border:1px solid {am(".25")};border-radius:6px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:{am(".7")};margin:2px;">☎ {ph}</span>'
        if lc: contacts += f'<span style="padding:3px 10px;background:{am(".08")};border:1px solid {am(".25")};border-radius:6px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:{am(".7")};margin:2px;">📍 {lc}</span>'
        if rd2.get("linkedin"):  contacts += '<span style="padding:3px 10px;background:rgba(10,102,194,.12);border:1px solid rgba(10,102,194,.3);border-radius:6px;font-size:10px;color:#6fa8dc;margin:2px;">🔗 LinkedIn</span>'
        if rd2.get("github"):    contacts += '<span style="padding:3px 10px;background:rgba(200,200,200,.08);border:1px solid rgba(200,200,200,.2);border-radius:6px;font-size:10px;color:#c8c8c8;margin:2px;">⌥ GitHub</span>'

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#13100a,#1a1508);
          border:1px solid {am('.3')};border-radius:16px;
          padding:22px 24px;margin-bottom:16px;display:flex;align-items:center;gap:20px;
          position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:1px;
            background:linear-gradient(90deg,transparent,#d4af5f,transparent);opacity:.5;"></div>
          <div style="width:60px;height:60px;border-radius:14px;flex-shrink:0;
            background:linear-gradient(135deg,#d4af5f,#f0cc70);
            display:flex;align-items:center;justify-content:center;
            font-family:'Libre Baskerville',serif;font-size:1.6rem;color:#0e0c07;">
            {initials}</div>
          <div>
            <div style="font-family:'Libre Baskerville',serif;font-size:1.4rem;
              color:#f5ead8;margin-bottom:3px;">{name_str}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
              color:{am('.45')};margin-bottom:9px;letter-spacing:.04em;">{doc_type}</div>
            <div style="display:flex;flex-wrap:wrap;gap:3px;">
              {contacts or f'<span style="color:{am(".35")};font-size:11px;">No contact info detected</span>'}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── summary + sentiment ───────────────────────────────────────────────────
    st.markdown(section_hdr("Core Analysis"), unsafe_allow_html=True)
    cs, csent = st.columns([3, 2])

    with cs:
        summary_text = safe_str(r.get("summary"), "No summary available.")
        body = (f'<div style="font-size:14px;color:{am(".85")};line-height:1.9;">{summary_text}</div>'
                f'<div style="display:inline-block;margin-top:14px;padding:4px 13px;'
                f'background:{am(".08")};border:1px solid {am(".25")};border-radius:20px;'
                f'font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#d4af5f;">'
                f'{doc_type}</div>')
        st.markdown(ds_card("📄","AI Summary", body), unsafe_allow_html=True)

    with csent:
        sent  = safe_str(r.get("sentiment"), "Neutral")
        sc2   = safe_dict(r.get("sentiment_scores"))
        pos   = round(safe_float(sc2.get("Positive")) * 100)
        neu   = round(safe_float(sc2.get("Neutral"))  * 100)
        neg   = round(safe_float(sc2.get("Negative")) * 100)
        scol  = {"Positive":"#6fcf97","Negative":"#eb5757","Neutral":"#f2c94c"}.get(sent,"#d4af5f")

        def sbar(lbl, pct, col):
            return (
                f'<div style="display:flex;justify-content:space-between;font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:9px;color:{am(".45")};margin-bottom:4px;letter-spacing:.04em;">'
                f'<span>{lbl}</span><span>{pct}%</span></div>'
                f'<div style="height:4px;background:{am(".08")};border-radius:2px;overflow:hidden;margin-bottom:10px;">'
                f'<div style="width:{pct}%;height:100%;background:{col};border-radius:2px;"></div></div>'
            )

        body = (f'<div style="font-family:\'Libre Baskerville\',serif;font-size:2.4rem;'
                f'color:{scol};line-height:1;margin-bottom:20px;">{sent}</div>'
                + sbar("Positive", pos, "#6fcf97")
                + sbar("Neutral",  neu, "#f2c94c")
                + sbar("Negative", neg, "#eb5757"))
        st.markdown(ds_card("◎","Sentiment", body, scol), unsafe_allow_html=True)

    # ── resume sections ───────────────────────────────────────────────────────
    if resume_mode and rd2:
        st.markdown(rule(), unsafe_allow_html=True)
        st.markdown(section_hdr("Resume Intelligence"), unsafe_allow_html=True)

        skills   = safe_list(rd2.get("skills"))
        sk_html  = (" ".join(amber_tag(s) for s in skills)
                    if skills else f'<span style="color:{am(".35")};font-size:13px;">No skills detected.</span>')
        st.markdown(ds_card("⚡","Skills Detected",
            f'<div style="display:flex;flex-wrap:wrap;gap:2px;">{sk_html}</div>'),
            unsafe_allow_html=True)

        def list_block(items, empty):
            if items:
                return "".join(sec_row(s) for s in items)
            return f'<div style="color:{am(".35")};font-size:12px;padding:8px 0;">{empty}</div>'

        c_p, c_e = st.columns(2)
        with c_p:
            st.markdown(ds_card("💻","Projects",
                list_block(safe_list(rd2.get("projects")),"No projects detected.")),
                unsafe_allow_html=True)
        with c_e:
            st.markdown(ds_card("💼","Experience",
                list_block(safe_list(rd2.get("experience")),"No experience detected.")),
                unsafe_allow_html=True)

        c_ed, c_ac = st.columns(2)
        with c_ed:
            st.markdown(ds_card("🎓","Education",
                list_block(safe_list(rd2.get("education")),"No education info.")),
                unsafe_allow_html=True)
        with c_ac:
            st.markdown(ds_card("🏆","Achievements",
                list_block(safe_list(rd2.get("achievements")),"No achievements detected.")),
                unsafe_allow_html=True)

        missing = safe_list(rd2.get("missing"))
        if missing:
            tags = " ".join(miss_tag(m) for m in missing)
            st.markdown(
                f'<div style="background:rgba(255,160,50,.05);border:1px solid rgba(255,160,50,.22);'
                f'border-radius:12px;padding:16px 18px;margin-bottom:12px;">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
                f'text-transform:uppercase;letter-spacing:.16em;color:#ffaa40;margin-bottom:10px;">'
                f'⚠ Missing fields — strengthen your profile</div>'
                f'<div>{tags}</div></div>',
                unsafe_allow_html=True,
            )

    # ── entities + keyphrases ─────────────────────────────────────────────────
    st.markdown(rule(), unsafe_allow_html=True)
    st.markdown(section_hdr("Entities & Key Phrases"), unsafe_allow_html=True)

    ent_html = ""
    for key, (lbl, fg, bg) in ENTITY_MAP.items():
        vals = safe_list(ents_raw.get(key))
        if not vals:
            continue
        tags = " ".join(colored_tag(safe_str(v), fg, bg) for v in vals)
        ent_html += (f'<div style="margin-bottom:13px;">'
                     f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
                     f'text-transform:uppercase;letter-spacing:.12em;color:{am(".4")};margin-bottom:6px;">'
                     f'{lbl}</div><div>{tags}</div></div>')
    if not ent_html:
        ent_html = f'<span style="color:{am(".35")};font-size:13px;">No entities detected.</span>'

    kp_html = (" ".join(kp_tag(safe_str(k)) for k in kp_list)
               if kp_list else f'<span style="color:{am(".35")};font-size:13px;">None detected.</span>')

    c_ent, c_kp = st.columns([3, 2])
    with c_ent:
        st.markdown(ds_card("◈","Named Entities",
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:8px;">'
            f'{ent_html}</div>'), unsafe_allow_html=True)
    with c_kp:
        st.markdown(ds_card("◇","Key Phrases",
            f'<div style="display:flex;flex-wrap:wrap;gap:2px;">{kp_html}</div>'),
            unsafe_allow_html=True)

    # ── readability ───────────────────────────────────────────────────────────
    st.markdown(rule(), unsafe_allow_html=True)
    st.markdown(section_hdr("Readability Analysis"), unsafe_allow_html=True)

    rd3   = safe_dict(stats.get("readability"))
    lang  = safe_str(stats.get("language"), "en").upper()
    lex   = safe_float(stats.get("lexical_diversity"))

    rcells = [
        (safe_str(rd3.get("flesch_reading_ease"),"-"),  "#d4af5f","Flesch Ease"),
        (safe_str(rd3.get("flesch_kincaid_grade"),"-"), "#f0cc70","FK Grade"),
        (safe_str(rd3.get("gunning_fog_index"),"-"),     "#ffaa40","Gunning Fog"),
        (safe_str(rd3.get("interpretation"),"-"),        "#a8d5a2","Interpretation"),
    ]
    rh = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;">'
    for val, col, lbl in rcells:
        rh += (f'<div style="background:{am(".04")};border:1px solid {am(".12")};'
               f'border-radius:10px;padding:14px;text-align:center;">'
               f'<div style="font-family:\'Libre Baskerville\',serif;font-size:1.7rem;'
               f'color:{col};line-height:1;margin-bottom:4px;">{val}</div>'
               f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
               f'text-transform:uppercase;letter-spacing:.1em;color:{am(".4")};">{lbl}</div>'
               f'</div>')
    rh += f'</div>'
    rh += (f'<div style="display:flex;gap:20px;font-family:\'IBM Plex Mono\',monospace;'
           f'font-size:10px;color:{am(".45")};padding-top:10px;border-top:1px solid {am(".08")};">'
           f'<span>Language: <strong style="color:{am(".7")};">{lang}</strong></span>'
           f'<span>Lex diversity: <strong style="color:{am(".7")};">{lex:.3f}</strong></span>'
           f'</div>')

    st.markdown(ds_card("📈","Readability Metrics", rh), unsafe_allow_html=True)

    # ── json + download ───────────────────────────────────────────────────────
    st.markdown(rule(), unsafe_allow_html=True)
    with st.expander("◈ View raw JSON response"):
        st.json(r)

    json_str = json.dumps(r, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇ Download full JSON result",
        data=json_str,
        file_name=Path(fn).stem + "_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

elif uploaded is None:
    st.markdown(f"""
    <div style="text-align:center;padding:5rem 2rem;
      border:1px dashed {am('.15')};border-radius:16px;margin-top:2rem;
      position:relative;">{corner_box()}
      <div style="font-size:3rem;opacity:.18;margin-bottom:1rem;">◈</div>
      <div style="font-family:'Libre Baskerville',serif;font-size:1.2rem;
        color:{am('.4')};margin-bottom:8px;">Awaiting document</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
        color:{am('.25')};letter-spacing:.1em;">
        PDF · DOCX · JPG · PNG · BMP · TIFF · WEBP
      </div>
    </div>""", unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:2.5rem 0 1rem;margin-top:5rem;
  border-top:1px solid {am('.12')};">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
    color:{am('.3')};letter-spacing:.1em;text-transform:uppercase;">
    DocSense AI &nbsp;·&nbsp; HCL GUVI BuildBridge Hackathon 2026 — Track 2 &nbsp;·&nbsp;
    <a href="https://github.com/Chirag0071/AI-Document-Analysis" target="_blank"
       style="color:{am('.45')};text-decoration:none;">GitHub ↗</a>
    &nbsp;·&nbsp;
    <a href="http://127.0.0.1:8000/docs" target="_blank"
       style="color:{am('.45')};text-decoration:none;">API Docs ↗</a>
  </div>
</div>""", unsafe_allow_html=True)