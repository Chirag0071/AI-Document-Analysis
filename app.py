"""
app.py — DocSense AI Streamlit Frontend
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

# ── Page config ───────────────────────────────────────────────────────────────
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400;1,600&display=swap');

html, body, [class*="css"] {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  background: #000005 !important;
  color: #d4ceff !important;
}

.stApp { background: #000005 !important; }

.block-container {
  padding: 2rem 3rem !important;
  max-width: 1100px !important;
  background: transparent !important;
}

/* Hero */
.hero-wrap {
  text-align: center;
  padding: 3.5rem 0 2.5rem;
  border-bottom: 1px solid rgba(180,100,255,0.12);
  margin-bottom: 2.5rem;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 18px;
  border: 1px solid rgba(184,71,255,0.25);
  border-radius: 30px;
  background: rgba(184,71,255,0.07);
  font-size: 11px;
  color: #b847ff;
  font-weight: 600;
  letter-spacing: 0.08em;
  margin-bottom: 24px;
}
.hero-title {
  font-family: 'Lora', serif !important;
  font-size: 3.2rem;
  color: #fff;
  line-height: 1.06;
  letter-spacing: -0.025em;
  margin-bottom: 16px;
}
.hero-em {
  font-style: italic;
  background: linear-gradient(120deg, #b847ff, #ff4dca);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-desc {
  color: #4a4570;
  font-size: 0.95rem;
  max-width: 500px;
  margin: 0 auto 32px;
  line-height: 1.75;
  font-weight: 300;
}
.tech-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
}
.tech-tag {
  padding: 5px 12px;
  border-radius: 8px;
  font-size: 11px;
  border: 1px solid rgba(255,255,255,0.07);
  background: rgba(255,255,255,0.025);
  color: rgba(255,255,255,0.38);
}

/* Cards */
.ds-card {
  background: #07091a;
  border: 1px solid rgba(180,100,255,0.12);
  border-radius: 18px;
  padding: 24px;
  margin-bottom: 14px;
  position: relative;
  overflow: hidden;
}
.ds-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, #b847ff 40%, #ff4dca 60%, transparent);
}
.card-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: rgba(255,255,255,0.35);
  margin-bottom: 14px;
}
.sum-text {
  font-size: 15px;
  color: rgba(255,255,255,0.8);
  line-height: 1.85;
  font-weight: 300;
}
.dtype-pill {
  display: inline-block;
  padding: 4px 12px;
  background: rgba(232,200,74,0.08);
  border: 1px solid rgba(232,200,74,0.2);
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  color: #e8c84a;
  margin-top: 12px;
}
.sent-pos { font-family: 'Lora', serif; font-size: 2.2rem; color: #00ffc2; }
.sent-neg { font-family: 'Lora', serif; font-size: 2.2rem; color: #ff3d5a; }
.sent-neu { font-family: 'Lora', serif; font-size: 2.2rem; color: #ff8c42; }
.met-num {
  font-family: 'Lora', serif;
  font-size: 2rem;
  color: #fff;
  line-height: 1;
}
.met-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #4a4570;
  margin-top: 4px;
  font-weight: 700;
}
.etag {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  margin: 2px;
}
.kpill {
  display: inline-block;
  padding: 6px 14px;
  background: rgba(184,71,255,0.08);
  border: 1px solid rgba(184,71,255,0.18);
  border-radius: 8px;
  font-size: 12px;
  color: #d4a0ff;
  margin: 3px;
}
.sbar-wrap { margin-bottom: 11px; }
.sbar-top {
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
}
.sbar-fill { height: 100%; border-radius: 3px; }
.res-ok {
  display: inline-block;
  padding: 5px 14px;
  background: rgba(0,255,194,0.09);
  border: 1px solid rgba(0,255,194,0.22);
  border-radius: 20px;
  font-size: 10px;
  font-weight: 700;
  color: #00ffc2;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* Streamlit overrides */
div[data-testid="stFileUploader"] {
  background: rgba(7,9,26,0.9) !important;
  border: 1px solid rgba(184,71,255,0.2) !important;
  border-radius: 16px !important;
  padding: 1rem !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: rgba(184,71,255,0.45) !important;
}
.stButton > button {
  background: linear-gradient(135deg, #b847ff, #ff4dca) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Bricolage Grotesque', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  padding: 0.75rem 2rem !important;
  width: 100% !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:disabled { opacity: 0.25 !important; }
.stProgress > div > div {
  background: linear-gradient(90deg, #b847ff, #ff4dca) !important;
}
.stTextInput > div > div > input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 9px !important;
  color: #fff !important;
  font-family: 'Bricolage Grotesque', sans-serif !important;
}
.stTextInput > div > div > input:focus {
  border-color: rgba(184,71,255,0.45) !important;
}
label, .stTextInput label {
  color: #4a4570 !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
}
.stExpander {
  background: #07091a !important;
  border: 1px solid rgba(180,100,255,0.12) !important;
  border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">&#9650; HCL GUVI BuildBridge Hackathon 2026 — Track 2</div>
  <div class="hero-title">Transform any document into <span class="hero-em">intelligence</span></div>
  <p class="hero-desc">Upload PDF, DOCX, or image up to 150 MB. The AI pipeline extracts summaries, named entities, sentiment, keyphrases and readability metrics instantly.</p>
  <div class="tech-row">
    <span class="tech-tag">Groq Llama 3.3 70B</span>
    <span class="tech-tag">spaCy NER</span>
    <span class="tech-tag">FinBERT Sentiment</span>
    <span class="tech-tag">Random Forest</span>
    <span class="tech-tag">VADER + TextBlob</span>
    <span class="tech-tag">YAKE Keyphrases</span>
    <span class="tech-tag">Tesseract OCR</span>
    <span class="tech-tag">Decision Tree Ensemble</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload + Config ───────────────────────────────────────────────────────────
col_up, col_cfg = st.columns([3, 2])

with col_up:
    uploaded = st.file_uploader(
        "Drop your document here (PDF, DOCX, JPG, PNG, BMP, TIFF, WEBP — max 150 MB)",
        type=["pdf","docx","doc","jpg","jpeg","png","bmp","tiff","webp"],
        label_visibility="visible",
    )

with col_cfg:
    api_url = st.text_input("API URL", value=API_URL)
    api_key = st.text_input("API Key", value=API_KEY, type="password")
    st.markdown(
        '<div style="font-size:11px;color:#4a4570;margin-top:4px;">&#9888; Keep defaults for local server.</div>',
        unsafe_allow_html=True
    )

# ── File info bar ─────────────────────────────────────────────────────────────
if uploaded:
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    ext = "." + uploaded.name.split(".")[-1].lower()
    ftype = EXT_MAP.get(ext, "unknown")

    if size_mb > MAX_MB:
        st.error(f"File too large: {size_mb:.1f} MB. Maximum is {MAX_MB} MB.")
        uploaded = None
    else:
        type_styles = {
            "pdf":   ("rgba(255,61,90,.12)",   "#ff8fa0", "rgba(255,61,90,.25)"),
            "docx":  ("rgba(184,71,255,.12)",  "#d4a0ff", "rgba(184,71,255,.25)"),
            "image": ("rgba(0,255,194,.08)",   "#80ffe3", "rgba(0,255,194,.15)"),
        }
        bg, fg, border = type_styles.get(ftype, ("rgba(100,100,100,.1)","#aaa","rgba(100,100,100,.2)"))
        icons = {"pdf":"📕","docx":"📘","image":"🖼️","unknown":"❓"}

        st.markdown(f"""
        <div style="background:rgba(184,71,255,0.06);border:1px solid rgba(184,71,255,0.18);
                    border-radius:13px;padding:13px 18px;margin:12px 0;
                    display:flex;align-items:center;gap:14px;">
          <span style="font-size:22px">{icons.get(ftype,'📄')}</span>
          <div style="flex:1">
            <div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:2px">{uploaded.name}</div>
            <div style="font-size:11px;color:#4a4570">{size_mb:.1f} MB &bull; {ftype.upper()} detected</div>
          </div>
          <span style="padding:3px 11px;border-radius:20px;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:.08em;
                       background:{bg};color:{fg};border:1px solid {border}">{ftype.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Analyze button ────────────────────────────────────────────────────────────
can_analyze = uploaded is not None and EXT_MAP.get("." + uploaded.name.split(".")[-1].lower(), "unknown") != "unknown"

if st.button("⚡ Analyze Document", disabled=not can_analyze):
    ext   = "." + uploaded.name.split(".")[-1].lower()
    ftype = EXT_MAP.get(ext, "unknown")

    steps = [
        "Reading file", "Extracting text", "Preprocessing",
        "Groq AI analysis", "spaCy NER", "Sentiment ensemble", "Computing stats"
    ]

    prog_bar  = st.progress(0)
    prog_text = st.empty()

    for i, step in enumerate(steps):
        prog_bar.progress(int((i + 1) / len(steps) * 80))
        prog_text.markdown(f"**{step}...**")
        time.sleep(0.25)

    try:
        b64 = base64.b64encode(uploaded.getvalue()).decode()
        resp = requests.post(
            f"{api_url}/api/document-analyze",
            json={"fileName": uploaded.name, "fileType": ftype, "fileBase64": b64},
            headers={"Content-Type": "application/json", "x-api-key": api_key},
            timeout=300,
        )
        resp.raise_for_status()
        r = resp.json()

        prog_bar.progress(100)
        prog_text.markdown("**✅ Analysis complete!**")
        time.sleep(0.4)
        prog_bar.empty()
        prog_text.empty()

        # ── Results ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:24px;padding-bottom:18px;border-bottom:1px solid rgba(180,100,255,0.12)">
          <div style="font-family:'Lora',serif;font-size:1.9rem;font-weight:600;color:#fff;letter-spacing:-.02em">Analysis Results</div>
          <span class="res-ok">&#10003; Complete</span>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        st_obj = r.get("document_stats", {})
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in [
            (m1, st_obj.get("word_count", "-"), "Words"),
            (m2, st_obj.get("sentence_count", "-"), "Sentences"),
            (m3, str(st_obj.get("reading_time_minutes", "-")) + " min", "Min read"),
            (m4, str(r.get("processing_time_seconds", "-")) + "s", "Process time"),
        ]:
            col.markdown(f"""
            <div class="ds-card" style="text-align:center;padding:20px 14px">
              <div class="met-num">{val}</div>
              <div class="met-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary + Sentiment
        cs, csent = st.columns([3, 2])

        with cs:
            st.markdown(f"""
            <div class="ds-card">
              <div class="card-label">AI Summary</div>
              <div class="sum-text">{r.get("summary","No summary available.")}</div>
              <div class="dtype-pill">{r.get("document_type","Unknown")}</div>
            </div>""", unsafe_allow_html=True)

        with csent:
            sent = r.get("sentiment", "Neutral")
            sc2  = r.get("sentiment_scores", {})
            pos  = round((sc2.get("Positive", 0)) * 100)
            neu  = round((sc2.get("Neutral",  0)) * 100)
            neg  = round((sc2.get("Negative", 0)) * 100)
            scls = "sent-pos" if sent=="Positive" else "sent-neg" if sent=="Negative" else "sent-neu"
            st.markdown(f"""
            <div class="ds-card">
              <div class="card-label">Sentiment</div>
              <div class="{scls}">{sent}</div>
              <div style="margin-top:18px">
                <div class="sbar-wrap">
                  <div class="sbar-top"><span>Positive</span><span>{pos}%</span></div>
                  <div class="sbar-track"><div class="sbar-fill" style="width:{pos}%;background:#00ffc2"></div></div>
                </div>
                <div class="sbar-wrap">
                  <div class="sbar-top"><span>Neutral</span><span>{neu}%</span></div>
                  <div class="sbar-track"><div class="sbar-fill" style="width:{neu}%;background:#ff8c42"></div></div>
                </div>
                <div class="sbar-wrap">
                  <div class="sbar-top"><span>Negative</span><span>{neg}%</span></div>
                  <div class="sbar-track"><div class="sbar-fill" style="width:{neg}%;background:#ff3d5a"></div></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Entities
        ents = r.get("entities", {})
        ent_cfg = {
            "names":         ("Names",         "rgba(6,182,212,.1)",   "#67e8f9"),
            "organizations": ("Organizations", "rgba(255,77,202,.1)",  "#f9a8d4"),
            "dates":         ("Dates",         "rgba(0,255,194,.08)",  "#80ffe3"),
            "locations":     ("Locations",     "rgba(251,191,36,.09)", "#fde68a"),
            "amounts":       ("Amounts",       "rgba(255,61,90,.1)",   "#fda4af"),
            "percentages":   ("Percentages",   "rgba(255,140,66,.1)",  "#fed7aa"),
            "emails":        ("Emails",        "rgba(184,71,255,.1)",  "#d4a0ff"),
            "phones":        ("Phones",        "rgba(100,200,80,.1)",  "#bbf7d0"),
            "urls":          ("URLs",          "rgba(99,102,241,.1)",  "#c7d2fe"),
        }
        tags_html = ""
        for k, (lbl, bg2, fg2) in ent_cfg.items():
            vals = ents.get(k, [])
            if not vals:
                continue
            tags_html += f'<div style="margin-bottom:12px"><div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#4a4570;margin-bottom:6px">{lbl}</div><div>'
            tags_html += "".join(f'<span class="etag" style="background:{bg2};color:{fg2}">{v}</span>' for v in vals)
            tags_html += "</div></div>"

        if not tags_html:
            tags_html = '<div style="color:#4a4570;font-size:13px">No entities detected.</div>'

        st.markdown(f"""
        <div class="ds-card">
          <div class="card-label">Named Entities</div>
          <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px">
            {tags_html}
          </div>
        </div>""", unsafe_allow_html=True)

        # Keyphrases + Readability
        ck, cr = st.columns(2)

        with ck:
            kp = r.get("key_phrases", [])
            pills = "".join(f'<span class="kpill">{k}</span>' for k in kp) if kp else '<span style="color:#4a4570;font-size:13px">None detected.</span>'
            st.markdown(f"""
            <div class="ds-card">
              <div class="card-label">Key Phrases</div>
              <div>{pills}</div>
            </div>""", unsafe_allow_html=True)

        with cr:
            rd2 = st_obj.get("readability", {})
            st.markdown(f"""
            <div class="ds-card">
              <div class="card-label">Readability Scores</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div style="background:rgba(255,255,255,.03);border-radius:10px;padding:13px">
                  <div style="font-family:'Lora',serif;font-size:1.6rem;font-weight:600;color:#b847ff;line-height:1;margin-bottom:3px">{rd2.get("flesch_reading_ease","-")}</div>
                  <div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a4570;font-weight:700">Flesch Ease</div>
                </div>
                <div style="background:rgba(255,255,255,.03);border-radius:10px;padding:13px">
                  <div style="font-family:'Lora',serif;font-size:1.6rem;font-weight:600;color:#ff4dca;line-height:1;margin-bottom:3px">{rd2.get("flesch_kincaid_grade","-")}</div>
                  <div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a4570;font-weight:700">FK Grade</div>
                </div>
                <div style="background:rgba(255,255,255,.03);border-radius:10px;padding:13px">
                  <div style="font-family:'Lora',serif;font-size:1.6rem;font-weight:600;color:#ff3d5a;line-height:1;margin-bottom:3px">{rd2.get("gunning_fog_index","-")}</div>
                  <div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a4570;font-weight:700">Gunning Fog</div>
                </div>
                <div style="background:rgba(255,255,255,.03);border-radius:10px;padding:13px">
                  <div style="font-size:1rem;font-weight:600;color:#00ffc2;margin-bottom:3px">{rd2.get("interpretation","-")}</div>
                  <div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a4570;font-weight:700">Interpretation</div>
                </div>
              </div>
              <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(180,100,255,0.12);font-size:11px;color:#4a4570;display:flex;gap:18px">
                <span>Language: <b style="color:rgba(255,255,255,.65)">{st_obj.get("language","en").upper()}</b></span>
                <span>Lexical div: <b style="color:rgba(255,255,255,.65)">{st_obj.get("lexical_diversity",0):.3f}</b></span>
              </div>
            </div>""", unsafe_allow_html=True)

        # JSON
        with st.expander("View raw JSON response"):
            st.json(r)

        # Download
        json_str = json.dumps(r, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Download full JSON result",
            data=json_str,
            file_name=Path(uploaded.name).stem + "_analysis.json",
            mime="application/json",
        )

    except requests.exceptions.ConnectionError:
        prog_bar.empty(); prog_text.empty()
        st.error("Cannot connect to API server.\nMake sure it is running:\nuvicorn main:app --reload")
    except Exception as e:
        prog_bar.empty(); prog_text.empty()
        st.error(f"Error: {str(e)}")

elif not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem">
      <div style="font-size:4rem;margin-bottom:1rem">&#8679;</div>
      <div style="color:#4a4570;font-size:14px">Upload a document above to get started</div>
      <div style="font-size:12px;color:#2a2550;margin-top:8px">PDF &bull; DOCX &bull; JPG &bull; PNG &bull; BMP &bull; TIFF &bull; WEBP</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
            border-top:1px solid rgba(180,100,255,0.12);
            color:#4a4570;font-size:12px">
  HCL GUVI BuildBridge Hackathon 2026 &mdash; Track 2: AI Document Analysis &nbsp;&bull;&nbsp;
  <a href="https://github.com/Chirag0071/AI-Document-Analysis" style="color:rgba(184,71,255,.7);text-decoration:none">GitHub</a>
  &nbsp;&bull;&nbsp;
  <a href="http://127.0.0.1:8000/docs" style="color:rgba(184,71,255,.7);text-decoration:none">API Docs</a>
</div>
""", unsafe_allow_html=True)