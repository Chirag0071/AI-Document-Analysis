app.py - Streamlit Frontend for DocSense AI
HCL GUVI BuildBridge Hackathon 2026 - Track 2
Run: streamlit run app.py
"""

import streamlit as st
import requests, base64, json, time, os
from pathlib import Path

st.set_page_config(page_title="DocSense AI", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "sk_track2_987654321")

EXT_MAP = {".pdf":"pdf",".docx":"docx",".doc":"docx",".jpg":"image",".jpeg":"image",".png":"image",".bmp":"image",".tiff":"image",".tif":"image",".webp":"image"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');
html, body, [class*="css"] { font-family: "Space Grotesk", sans-serif !important; }
.stApp { background: #03050a !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1100px; }

/* Hero */
.hero-wrap { text-align: center; padding: 3rem 0 2rem; border-bottom: 1px solid #1a2540; margin-bottom: 2rem; }
.hero-badge { display: inline-block; padding: 5px 18px; background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.25); border-radius: 20px; font-size: 12px; color: #c084fc; letter-spacing: 0.08em; margin-bottom: 20px; }
.hero-title { font-family: "Instrument Serif", serif !important; font-size: 3.5rem; color: #fff; line-height: 1.05; margin-bottom: 12px; }
.hero-title em { font-style: italic; color: #4f8ef7; }
.hero-desc { color: #4a5568; font-size: 0.95rem; max-width: 480px; margin: 0 auto; line-height: 1.7; }

/* Result cards */
.r-card { background: #0c1220; border: 1px solid #1a2540; border-radius: 16px; padding: 22px; margin-bottom: 14px; }
.r-card-top { position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #4f8ef7, #a855f7); border-radius: 16px 16px 0 0; }
.r-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; color: #4a5568; margin-bottom: 12px; }
.r-sum { font-size: 15px; color: rgba(255,255,255,0.8); line-height: 1.78; font-weight: 300; }
.sent-pos { font-family: "Instrument Serif", serif; font-size: 2.2rem; color: #10b981; }
.sent-neg { font-family: "Instrument Serif", serif; font-size: 2.2rem; color: #ef4444; }
.sent-neu { font-family: "Instrument Serif", serif; font-size: 2.2rem; color: #f59e0b; }
.met-num { font-family: "Instrument Serif", serif; font-size: 2rem; color: #4f8ef7; line-height: 1; }
.met-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #4a5568; margin-top: 4px; }
.dtype-pill { display: inline-block; padding: 3px 12px; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.25); border-radius: 20px; font-size: 12px; color: #f59e0b; font-weight: 600; }
.etag { display: inline-block; padding: 3px 10px; border-radius: 7px; font-size: 12px; font-weight: 500; margin: 2px; }
.kpill { display: inline-block; padding: 6px 14px; background: rgba(79,142,247,0.08); border: 1px solid rgba(79,142,247,0.2); border-radius: 8px; font-size: 13px; color: #93c5fd; margin: 3px; }

/* Upload */
div[data-testid="stFileUploader"] { background: rgba(12,18,32,0.8); border: 1px solid #1a2540 !important; border-radius: 16px !important; padding: 1rem !important; }
div[data-testid="stFileUploader"]:hover { border-color: rgba(79,142,247,0.4) !important; }

/* Button */
.stButton > button { background: #4f8ef7 !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: "Space Grotesk", sans-serif !important; font-weight: 600 !important; font-size: 15px !important; padding: 0.7rem 2rem !important; width: 100% !important; transition: opacity 0.2s !important; }
.stButton > button:hover { opacity: 0.88 !important; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #4f8ef7, #a855f7) !important; }

/* Inputs */
.stTextInput > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid #1a2540 !important; border-radius: 10px !important; color: #fff !important; font-family: "Space Grotesk", sans-serif !important; }
.stTextInput > div > div > input:focus { border-color: rgba(79,142,247,0.5) !important; }
label { color: #4a5568 !important; font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">&#9889; Track 2: AI-Powered Document Analysis &mdash; HCL GUVI BuildBridge 2026</div>
  <div class="hero-title">Intelligent <em>Document</em> Analysis</div>
  <p class="hero-desc">Upload any PDF, DOCX, or image. Get AI-powered summary, named entities, sentiment analysis, keyphrases and readability scores instantly.</p>
</div>
""", unsafe_allow_html=True)

# Upload row
col_up, col_cfg = st.columns([3, 2])

with col_up:
    uploaded = st.file_uploader("Upload your document", type=["pdf","docx","doc","jpg","jpeg","png","bmp","tiff","webp"], label_visibility="collapsed")

with col_cfg:
    api_url = st.text_input("API URL", value=API_URL)
    api_key = st.text_input("API Key", value=API_KEY, type="password")

# File info
if uploaded:
    ext = "." + uploaded.name.split(".")[-1].lower()
    ftype = EXT_MAP.get(ext, "unknown")
    size = len(uploaded.getvalue()) / 1024
    type_colors = {"pdf": "#f87171", "docx": "#93c5fd", "image": "#d8b4fe", "unknown": "#9ca3af"}
    color = type_colors.get(ftype, "#9ca3af")
    st.markdown(f"""
    <div style="background:rgba(79,142,247,0.06);border:1px solid rgba(79,142,247,0.18);
                border-radius:12px;padding:14px 18px;margin:12px 0;
                display:flex;align-items:center;gap:16px;">
      <span style="font-size:24px">{"📕" if ftype=="pdf" else "📘" if ftype=="docx" else "🖼️"}</span>
      <div style="flex:1">
        <div style="font-size:14px;font-weight:600;color:#fff">{uploaded.name}</div>
        <div style="font-size:12px;color:#4a5568;margin-top:2px">{size:.1f} KB &bull; Detected as {ftype.upper()}</div>
      </div>
      <span style="padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;
                   background:rgba({",".join(str(int(color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.15);
                   color:{color};border:1px solid {color}44">{ftype.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

# Analyze
if st.button("⚡ Analyze Document", disabled=(not uploaded or EXT_MAP.get("." + uploaded.name.split(".")[-1].lower(), "unknown") == "unknown")):
    ext = "." + uploaded.name.split(".")[-1].lower()
    ftype = EXT_MAP.get(ext, "unknown")
    b64 = base64.b64encode(uploaded.getvalue()).decode()

    steps = ["Reading file","Extracting text","Preprocessing","Groq AI analysis","NER extraction","Sentiment analysis","Computing stats"]
    prog = st.progress(0)
    status = st.empty()

    for i, step in enumerate(steps):
        prog.progress(int((i+1)/len(steps)*80))
        status.markdown(f"**{step}...**")
        time.sleep(0.25)

    try:
        resp = requests.post(f"{api_url}/api/document-analyze",
            json={"fileName": uploaded.name, "fileType": ftype, "fileBase64": b64},
            headers={"Content-Type":"application/json","x-api-key":api_key}, timeout=300)
        resp.raise_for_status()
        r = resp.json()

        prog.progress(100)
        status.markdown("**✅ Analysis complete!**")
        time.sleep(0.4)
        prog.empty(); status.empty()

        st.markdown("---")
        st.markdown("<div class='hero-title' style='font-size:2rem;margin-bottom:1.5rem'>Analysis Results</div>", unsafe_allow_html=True)

        st_obj = r.get("document_stats", {})
        m1,m2,m3,m4 = st.columns(4)
        for col, num, lbl in [(m1,st_obj.get("word_count","-"),"Words"),(m2,st_obj.get("sentence_count","-"),"Sentences"),(m3,str(st_obj.get("reading_time_minutes","-"))+" min","Min read"),(m4,str(r.get("processing_time_seconds","-"))+"s","Process time")]:
            col.markdown(f'<div class="r-card" style="text-align:center"><div class="met-num">{num}</div><div class="met-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cs, csent = st.columns([3,2])

        with cs:
            st.markdown(f"""<div class="r-card">
              <div class="r-label">AI Generated Summary</div>
              <div class="r-sum">{r.get("summary","No summary.")}</div>
              <div style="margin-top:14px;padding-top:12px;border-top:1px solid #1a2540;font-size:12px;color:#4a5568">
                Document type: <span class="dtype-pill">{r.get("document_type","Unknown")}</span>
              </div></div>""", unsafe_allow_html=True)

        with csent:
            sent = r.get("sentiment","Neutral")
            sc2 = r.get("sentiment_scores",{})
            pos = round((sc2.get("Positive",0))*100)
            neu = round((sc2.get("Neutral",0))*100)
            neg = round((sc2.get("Negative",0))*100)
            cls = "sent-pos" if sent=="Positive" else "sent-neg" if sent=="Negative" else "sent-neu"
            st.markdown(f"""<div class="r-card">
              <div class="r-label">Sentiment Analysis</div>
              <div class="{cls}">{sent}</div>
              <div style="margin-top:16px">
                <div style="display:flex;justify-content:space-between;font-size:12px;color:#4a5568;margin-bottom:4px"><span>Positive</span><span>{pos}%</span></div>
                <div style="background:rgba(255,255,255,0.06);border-radius:3px;height:5px;overflow:hidden;margin-bottom:10px"><div style="width:{pos}%;background:#10b981;height:100%;border-radius:3px"></div></div>
                <div style="display:flex;justify-content:space-between;font-size:12px;color:#4a5568;margin-bottom:4px"><span>Neutral</span><span>{neu}%</span></div>
                <div style="background:rgba(255,255,255,0.06);border-radius:3px;height:5px;overflow:hidden;margin-bottom:10px"><div style="width:{neu}%;background:#f59e0b;height:100%;border-radius:3px"></div></div>
                <div style="display:flex;justify-content:space-between;font-size:12px;color:#4a5568;margin-bottom:4px"><span>Negative</span><span>{neg}%</span></div>
                <div style="background:rgba(255,255,255,0.06);border-radius:3px;height:5px;overflow:hidden"><div style="width:{neg}%;background:#ef4444;height:100%;border-radius:3px"></div></div>
              </div></div>""", unsafe_allow_html=True)

        ents = r.get("entities",{})
        ent_cfg = {"names":("Names","rgba(6,182,212,0.1)","#67e8f9"),"organizations":("Organizations","rgba(168,85,247,0.1)","#d8b4fe"),"dates":("Dates","rgba(16,185,129,0.1)","#6ee7b7"),"locations":("Locations","rgba(245,158,11,0.1)","#fcd34d"),"amounts":("Amounts","rgba(239,68,68,0.1)","#fca5a5"),"percentages":("Percentages","rgba(249,115,22,0.1)","#fdba74"),"emails":("Emails","rgba(79,142,247,0.1)","#93c5fd"),"phones":("Phones","rgba(132,204,22,0.1)","#bef264"),"urls":("URLs","rgba(99,102,241,0.1)","#c7d2fe")}
        tags_html = ""
        for k,(lbl,bg,fg) in ent_cfg.items():
            vals = ents.get(k,[])
            if vals:
                tags_html += f'<div style="margin-bottom:12px"><div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#4a5568;margin-bottom:6px">{lbl}</div><div>' + "".join(f'<span class="etag" style="background:{bg};color:{fg}">{v}</span>' for v in vals) + "</div></div>"
        if not tags_html: tags_html = '<div style="color:#4a5568;font-size:13px">No entities detected.</div>'
        st.markdown(f'<div class="r-card"><div class="r-label">Named Entities</div><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px">{tags_html}</div></div>', unsafe_allow_html=True)

        ck, cr = st.columns(2)
        kp = r.get("key_phrases",[])
        with ck:
            pills = "".join(f'<span class="kpill">{k}</span>' for k in kp) if kp else '<span style="color:#4a5568">None detected.</span>'
            st.markdown(f'<div class="r-card"><div class="r-label">Key Phrases</div><div>{pills}</div></div>', unsafe_allow_html=True)
        with cr:
            rd2 = st_obj.get("readability",{})
            st.markdown(f"""<div class="r-card">
              <div class="r-label">Readability Scores</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:12px"><div style="font-family:'Instrument Serif',serif;font-size:1.6rem;color:#4f8ef7">{rd2.get("flesch_reading_ease","-")}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a5568">Flesch Ease</div></div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:12px"><div style="font-family:'Instrument Serif',serif;font-size:1.6rem;color:#a855f7">{rd2.get("flesch_kincaid_grade","-")}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a5568">FK Grade</div></div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:12px"><div style="font-family:'Instrument Serif',serif;font-size:1.6rem;color:#ef4444">{rd2.get("gunning_fog_index","-")}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a5568">Gunning Fog</div></div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:12px"><div style="font-size:1rem;color:#10b981;font-weight:600">{rd2.get("interpretation","-")}</div><div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#4a5568">Interpretation</div></div>
              </div>
              <div style="margin-top:12px;padding-top:10px;border-top:1px solid #1a2540;font-size:12px;color:#4a5568">
                Language: <b style="color:rgba(255,255,255,0.65)">{st_obj.get("language","en").upper()}</b>
                &nbsp;&bull;&nbsp; Lexical diversity: <b style="color:rgba(255,255,255,0.65)">{st_obj.get("lexical_diversity",0):.3f}</b>
              </div></div>""", unsafe_allow_html=True)

        with st.expander("View raw JSON response"):
            st.json(r)

        json_str = json.dumps(r, indent=2, ensure_ascii=False)
        st.download_button("⬇️ Download full JSON result", json_str, file_name=Path(uploaded.name).stem+"_analysis.json", mime="application/json")

    except requests.exceptions.ConnectionError:
        prog.empty(); status.empty()
        st.error("Cannot connect to API server. Make sure it is running:\nuvicorn main:app --reload")
    except Exception as e:
        prog.empty(); status.empty()
        st.error(f"Error: {str(e)}")
else:
    if not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#1e293b">
          <div style="font-size:4rem;margin-bottom:1rem">&#8679;</div>
          <div style="color:#4a5568">Upload a document above to get started</div>
          <div style="font-size:13px;color:#1e293b;margin-top:6px">PDF &bull; DOCX &bull; JPG &bull; PNG &bull; BMP &bull; TIFF &bull; WEBP</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1a2540;color:#4a5568;font-size:12px">
  HCL GUVI BuildBridge Hackathon 2026 &mdash; Track 2: AI Document Analysis &nbsp;&bull;&nbsp;
  <a href="https://github.com/Chirag0071/AI-Document-Analysis" style="color:rgba(79,142,247,0.7);text-decoration:none">GitHub</a>
</div>
""", unsafe_allow_html=True)
