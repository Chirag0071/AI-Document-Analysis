"""
app.py — Streamlit Frontend for AI Document Analysis API
HCL GUVI BuildBridge Hackathon 2026

Run: streamlit run app.py
"""

import streamlit as st
import requests
import base64
import json
import time
import os
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "sk_track2_987654321")

SUPPORTED_TYPES = {
    "pdf":  [".pdf"],
    "docx": [".docx", ".doc"],
    "image":[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"],
}

EXT_TO_TYPE = {
    ext: ftype
    for ftype, exts in SUPPORTED_TYPES.items()
    for ext in exts
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.main { background: #0a0a0f; }

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f1a 100%);
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.hero-sub {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-family: 'DM Sans', sans-serif;
}

.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.sentiment-positive {
    color: #00ff88;
    font-weight: 700;
    font-size: 1.4rem;
}
.sentiment-negative {
    color: #ff4444;
    font-weight: 700;
    font-size: 1.4rem;
}
.sentiment-neutral {
    color: #ffaa00;
    font-weight: 700;
    font-size: 1.4rem;
}

.entity-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 2px;
    font-weight: 500;
}

.metric-box {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
}

.metric-label {
    color: #666;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-pdf   { background: rgba(255,100,100,0.15); color: #ff6464; border: 1px solid rgba(255,100,100,0.3); }
.badge-docx  { background: rgba(100,150,255,0.15); color: #6496ff; border: 1px solid rgba(100,150,255,0.3); }
.badge-image { background: rgba(150,100,255,0.15); color: #9664ff; border: 1px solid rgba(150,100,255,0.3); }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #555;
    margin-bottom: 0.5rem;
}

.keyphrase-pill {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(123, 47, 255, 0.15);
    border: 1px solid rgba(123, 47, 255, 0.3);
    border-radius: 20px;
    font-size: 0.82rem;
    color: #a078ff;
    margin: 3px;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(0,212,255,0.3);
    border-radius: 16px;
    padding: 1rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.6);
    background: rgba(0,212,255,0.03);
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7b2fff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #00d4ff, #7b2fff) !important;
}

.sidebar-info {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return EXT_TO_TYPE.get(ext, "unknown")


def get_file_type_badge(ftype: str) -> str:
    badges = {
        "pdf":   '<span class="badge badge-pdf">PDF</span>',
        "docx":  '<span class="badge badge-docx">DOCX</span>',
        "image": '<span class="badge badge-image">IMAGE</span>',
    }
    return badges.get(ftype, "")


def call_api(file_bytes: bytes, file_name: str, file_type: str) -> dict:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    payload = {
        "fileName": file_name,
        "fileType": file_type,
        "fileBase64": b64,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
    }
    response = requests.post(
        f"{API_URL}/api/document-analyze",
        json=payload,
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def sentiment_color(label: str) -> str:
    return {"Positive": "#00ff88", "Negative": "#ff4444", "Neutral": "#ffaa00"}.get(label, "#888")


def entity_color(key: str) -> str:
    colors = {
        "names":         ("rgba(0,212,255,0.15)",   "#00d4ff"),
        "organizations": ("rgba(123,47,255,0.15)",  "#9b5fff"),
        "dates":         ("rgba(0,255,136,0.15)",   "#00ff88"),
        "locations":     ("rgba(255,170,0,0.15)",   "#ffaa00"),
        "amounts":       ("rgba(255,107,107,0.15)", "#ff6b6b"),
        "percentages":   ("rgba(255,107,107,0.12)", "#ff8888"),
        "emails":        ("rgba(100,200,255,0.15)", "#64c8ff"),
        "phones":        ("rgba(180,255,100,0.15)", "#b4ff64"),
        "urls":          ("rgba(200,150,255,0.15)", "#c896ff"),
    }
    bg, fg = colors.get(key, ("rgba(100,100,100,0.15)", "#888"))
    return f"background:{bg};color:{fg};border:1px solid {fg}33;"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                    background:linear-gradient(90deg,#00d4ff,#7b2fff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
            📄 DocAnalyzer
        </div>
        <div style="color:#555;font-size:0.75rem;">HCL GUVI BuildBridge 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")
    api_url_input = st.text_input("API URL", value=API_URL)
    api_key_input = st.text_input("API Key", value=API_KEY, type="password")

    st.markdown("---")
    st.markdown("### 📋 Supported Formats")

    for ftype, exts in SUPPORTED_TYPES.items():
        st.markdown(f"""
        <div class="sidebar-info">
            <b style="text-transform:uppercase;font-size:0.8rem;">{ftype}</b><br>
            <span style="color:#555;font-size:0.8rem;">{' · '.join(exts)}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Pipeline")
    pipeline_steps = [
        ("🔍", "Text Extraction", "PDF/DOCX/Image OCR"),
        ("🧹", "Preprocessing", "Clean & normalize"),
        ("🤖", "Groq LLM", "Summary + NER + Sentiment"),
        ("🏷️", "spaCy NER", "Entity extraction"),
        ("😊", "4-Model Sentiment", "VADER+TextBlob+BERT+DT"),
        ("📊", "RF Classifier", "Document type (10 types)"),
        ("🔑", "YAKE+TF-IDF", "Keyphrase extraction"),
        ("📈", "Readability", "Flesch + Fog + FK"),
    ]
    for icon, title, desc in pipeline_steps:
        st.markdown(f"""
        <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:8px;">
            <span style="font-size:1rem;">{icon}</span>
            <div>
                <div style="font-size:0.8rem;font-weight:600;color:#ccc;">{title}</div>
                <div style="font-size:0.72rem;color:#555;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Health check
    st.markdown("---")
    if st.button("🔌 Check API Health"):
        try:
            r = requests.get(f"{api_url_input}/health", timeout=5)
            if r.status_code == 200:
                st.success("✅ API is online!")
            else:
                st.error(f"❌ Status: {r.status_code}")
        except Exception:
            st.error("❌ Cannot connect to API")


# ── Main Page ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-title">AI Document Analyzer</div>
<div class="hero-sub">
    Extract • Summarize • Classify • Analyze — any document in seconds
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf", "docx", "doc", "jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Supports PDF, DOCX, JPG, PNG, BMP, TIFF, WEBP",
    )

with col_info:
    if uploaded_file:
        ftype = detect_file_type(uploaded_file.name)
        fsize = len(uploaded_file.getvalue()) / 1024

        st.markdown(f"""
        <div class="result-card" style="height:100%;">
            <div class="section-title">Detected File</div>
            <div style="margin-bottom:8px;">{get_file_type_badge(ftype)}</div>
            <div style="font-size:0.9rem;color:#ccc;margin-bottom:4px;">
                📄 <b>{uploaded_file.name}</b>
            </div>
            <div style="font-size:0.8rem;color:#666;">
                Size: {fsize:.1f} KB
            </div>
            <div style="font-size:0.8rem;color:#666;">
                Type detected: <b style="color:#00d4ff;">{ftype.upper()}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📂</div>
            <div style="color:#444;font-size:0.85rem;">Upload a file to see details</div>
        </div>
        """, unsafe_allow_html=True)

# ── Analyze Button ────────────────────────────────────────────────────────────

if uploaded_file:
    if st.button("🚀 Analyze Document"):
        ftype = detect_file_type(uploaded_file.name)

        if ftype == "unknown":
            st.error("Unsupported file type. Please upload PDF, DOCX, or image.")
        else:
            # Progress animation
            progress_bar = st.progress(0)
            status_text  = st.empty()

            steps = [
                (10, "📥 Reading file..."),
                (25, "🔍 Extracting text..."),
                (45, "🧹 Preprocessing..."),
                (60, "🤖 Running Groq AI analysis..."),
                (75, "🏷️ Extracting entities..."),
                (88, "😊 Analyzing sentiment..."),
                (95, "📊 Computing statistics..."),
                (100, "✅ Done!"),
            ]

            for pct, msg in steps[:-1]:
                progress_bar.progress(pct)
                status_text.markdown(f"**{msg}**")
                time.sleep(0.3)

            try:
                file_bytes = uploaded_file.getvalue()
                result = call_api(
                    file_bytes,
                    uploaded_file.name,
                    ftype,
                )
                progress_bar.progress(100)
                status_text.markdown("**✅ Analysis complete!**")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                # ── Results ───────────────────────────────────────────────────

                st.markdown("---")
                st.markdown("""
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;
                            font-weight:800;color:#fff;margin-bottom:1rem;">
                    📊 Analysis Results
                </div>
                """, unsafe_allow_html=True)

                # Top metrics row
                m1, m2, m3, m4 = st.columns(4)
                stats = result.get("document_stats", {})
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-number">{stats.get('word_count', 0)}</div>
                        <div class="metric-label">Words</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-number">{stats.get('sentence_count', 0)}</div>
                        <div class="metric-label">Sentences</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-number">{stats.get('reading_time_minutes', 0)}</div>
                        <div class="metric-label">Min Read</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-number">{result.get('processing_time_seconds', 0)}s</div>
                        <div class="metric-label">Process Time</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 1: Summary + Sentiment
                col_sum, col_sent = st.columns([3, 2])

                with col_sum:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="section-title">📝 AI Summary</div>
                        <div style="font-size:0.95rem;color:#ddd;line-height:1.7;">
                            {result.get('summary', 'No summary available')}
                        </div>
                        <div style="margin-top:0.8rem;">
                            <span style="font-size:0.78rem;color:#555;">Document Type: </span>
                            <span style="font-size:0.82rem;color:#7b2fff;font-weight:600;">
                                {result.get('document_type', 'Unknown')}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_sent:
                    sentiment = result.get("sentiment", "Neutral")
                    scores    = result.get("sentiment_scores", {})
                    scolor    = sentiment_color(sentiment)
                    pos = scores.get("Positive", 0)
                    neu = scores.get("Neutral",  0)
                    neg = scores.get("Negative", 0)

                    st.markdown(f"""
                    <div class="result-card" style="height:100%;">
                        <div class="section-title">😊 Sentiment Analysis</div>
                        <div class="sentiment-{sentiment.lower()}" style="margin-bottom:1rem;">
                            {sentiment}
                        </div>
                        <div style="margin-bottom:0.4rem;">
                            <div style="display:flex;justify-content:space-between;
                                        font-size:0.78rem;color:#555;margin-bottom:2px;">
                                <span>Positive</span><span>{pos:.0%}</span>
                            </div>
                            <div style="background:#1a1a2e;border-radius:4px;height:6px;">
                                <div style="width:{pos*100:.0f}%;background:#00ff88;
                                            height:6px;border-radius:4px;"></div>
                            </div>
                        </div>
                        <div style="margin-bottom:0.4rem;">
                            <div style="display:flex;justify-content:space-between;
                                        font-size:0.78rem;color:#555;margin-bottom:2px;">
                                <span>Neutral</span><span>{neu:.0%}</span>
                            </div>
                            <div style="background:#1a1a2e;border-radius:4px;height:6px;">
                                <div style="width:{neu*100:.0f}%;background:#ffaa00;
                                            height:6px;border-radius:4px;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display:flex;justify-content:space-between;
                                        font-size:0.78rem;color:#555;margin-bottom:2px;">
                                <span>Negative</span><span>{neg:.0%}</span>
                            </div>
                            <div style="background:#1a1a2e;border-radius:4px;height:6px;">
                                <div style="width:{neg*100:.0f}%;background:#ff4444;
                                            height:6px;border-radius:4px;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Row 2: Entities
                st.markdown(f"""
                <div class="result-card">
                    <div class="section-title">🏷️ Named Entities</div>
                """, unsafe_allow_html=True)

                entities = result.get("entities", {})
                entity_labels = {
                    "names":         "👤 Names",
                    "organizations": "🏢 Organizations",
                    "dates":         "📅 Dates",
                    "locations":     "📍 Locations",
                    "amounts":       "💰 Amounts",
                    "percentages":   "📊 Percentages",
                    "emails":        "📧 Emails",
                    "phones":        "📞 Phones",
                    "urls":          "🔗 URLs",
                }

                ent_cols = st.columns(3)
                ent_items = [(k, v) for k, v in entities.items() if v]

                if ent_items:
                    for i, (key, values) in enumerate(ent_items):
                        col = ent_cols[i % 3]
                        with col:
                            tags_html = "".join(
                                f'<span class="entity-tag" style="{entity_color(key)}">{v}</span>'
                                for v in values
                            )
                            st.markdown(f"""
                            <div style="margin-bottom:0.8rem;">
                                <div style="font-size:0.72rem;color:#555;
                                            margin-bottom:4px;font-weight:600;">
                                    {entity_labels.get(key, key.upper())}
                                </div>
                                <div>{tags_html}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div style="color:#444;font-size:0.85rem;">No entities detected</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

                # Row 3: Keyphrases + Readability
                col_kp, col_read = st.columns([3, 2])

                with col_kp:
                    keyphrases = result.get("key_phrases", [])
                    pills = "".join(
                        f'<span class="keyphrase-pill">{kp}</span>'
                        for kp in keyphrases
                    )
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="section-title">🔑 Key Phrases</div>
                        <div>{pills if pills else
                              '<span style="color:#444;">None detected</span>'}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_read:
                    readability = stats.get("readability", {})
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="section-title">📈 Readability</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                            <div>
                                <div style="font-size:0.7rem;color:#555;">Flesch Ease</div>
                                <div style="font-size:1.1rem;font-weight:700;color:#00d4ff;">
                                    {readability.get('flesch_reading_ease', 0)}
                                </div>
                            </div>
                            <div>
                                <div style="font-size:0.7rem;color:#555;">FK Grade</div>
                                <div style="font-size:1.1rem;font-weight:700;color:#7b2fff;">
                                    {readability.get('flesch_kincaid_grade', 0)}
                                </div>
                            </div>
                            <div>
                                <div style="font-size:0.7rem;color:#555;">Gunning Fog</div>
                                <div style="font-size:1.1rem;font-weight:700;color:#ff6b6b;">
                                    {readability.get('gunning_fog_index', 0)}
                                </div>
                            </div>
                            <div>
                                <div style="font-size:0.7rem;color:#555;">Interpretation</div>
                                <div style="font-size:0.85rem;font-weight:600;color:#ffaa00;">
                                    {readability.get('interpretation', '?')}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top:0.8rem;font-size:0.78rem;color:#555;">
                            Language: <b style="color:#ccc;">
                                {stats.get('language', 'en').upper()}
                            </b> &nbsp;|&nbsp;
                            Lexical diversity: <b style="color:#ccc;">
                                {stats.get('lexical_diversity', 0):.3f}
                            </b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Raw JSON expander
                with st.expander("🔧 View Raw JSON Response"):
                    st.json(result)

                # Download JSON button
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="⬇️ Download Full JSON Result",
                    data=json_str,
                    file_name=f"{Path(uploaded_file.name).stem}_analysis.json",
                    mime="application/json",
                )

            except requests.exceptions.ConnectionError:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ Cannot connect to API. Make sure server is running: `uvicorn main:app --reload`")
            except requests.exceptions.HTTPError as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ API Error {e.response.status_code}: {e.response.text[:300]}")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error: {str(e)}")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#333;">
        <div style="font-size:4rem;margin-bottom:1rem;">⬆️</div>
        <div style="font-size:1rem;color:#555;">Upload a document above to get started</div>
        <div style="font-size:0.85rem;color:#3a3a3a;margin-top:0.5rem;">
            PDF · DOCX · JPG · PNG · BMP · TIFF · WEBP
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1rem;
            border-top:1px solid #1a1a2e;color:#333;font-size:0.78rem;">
    HCL GUVI BuildBridge Hackathon 2026 — Track 2: AI-Powered Document Analysis
    &nbsp;|&nbsp;
    <a href="https://github.com/Chirag0071/AI-Document-Analysis"
       style="color:#7b2fff;text-decoration:none;">GitHub</a>
    &nbsp;|&nbsp;
    <a href="http://127.0.0.1:8000/docs" style="color:#00d4ff;text-decoration:none;">
        API Docs
    </a>
</div>
""", unsafe_allow_html=True)
