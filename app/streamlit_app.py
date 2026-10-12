"""
🌐 Advanced Multilingual RAG Chatbot — v4.0
A production-grade Streamlit app for multilingual document Q&A.

Author: A_srt343
Instagram: @project_srt343
"""

import sys
import os
import time
import json
import re
import html
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Generator

# ═══════════════════════════════════════════════════════════
#   🔧 Project root on sys.path
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    build_vectorstore,
    get_retriever,
    get_vectorstore_info,
    get_disk_usage,
    list_collections,
    delete_all_data,
    backup_vectorstore,
    restore_vectorstore,
    benchmark_embedding_model,
    EMBEDDING_MODELS,
    DEFAULT_EMBED_MODEL,
)
from llm import (
    ask_question,
    ask_question_stream,
    multi_query_retrieve,
    reformulate_question,
    compute_confidence,
    format_citations,
    detect_language,
    get_token_usage,
    reset_token_usage,
    clear_response_cache,
    AVAILABLE_MODELS,
    PRICING,
    STYLE_MODIFIERS,
    PERSONAS,
    estimate_cost,
)

# ═══════════════════════════════════════════════════════════
#   ⚙️ Environment
# ═══════════════════════════════════════════════════════════
load_dotenv()

# ═══════════════════════════════════════════════════════════
#   🎨 Page config
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nexus RAG · Multilingual AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Nexus RAG v4.0 — Multilingual document intelligence powered by GPT & Chroma.",
    },
)

# ═══════════════════════════════════════════════════════════
#   🗂️ Session state initialization
# ═══════════════════════════════════════════════════════════
def init_state() -> None:
    """Initialize all session state keys with safe defaults."""
    defaults: Dict[str, Any] = {
        "messages": [],
        "documents": {},
        "active_doc": None,
        "total_questions": 0,
        "search_history": [],
        "pinned_answers": [],
        "last_question": None,
        "pending_chip_question": None,
        "onboarding_seen": False,
        "theme": "light",
        "sidebar_section": "chat",
        "show_debug": False,
        "auto_scroll": True,
        "suggestion_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ═══════════════════════════════════════════════════════════
#   🎨 Theme system (light + dark)
# ═══════════════════════════════════════════════════════════
def get_theme_vars(theme: str) -> Dict[str, str]:
    """Return CSS variables for the chosen theme."""
    if theme == "dark":
        return {
            "bg": "#0f0f1a",
            "bg2": "#16162a",
            "surface": "#1a1a2e",
            "surface2": "#22223b",
            "border": "#2d2d4a",
            "text": "#e8e8f0",
            "text-muted": "#9a9ab0",
            "primary": "#8b5cf6",
            "primary2": "#6366f1",
            "accent": "#ec4899",
            "success": "#10b981",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "shadow": "rgba(0,0,0,0.4)",
            "header-grad": "linear-gradient(135deg, #4c1d95 0%, #831843 50%, #1e3a8a 100%)",
        }
    return {
        "bg": "#f7f8fc",
        "bg2": "#eef1f8",
        "surface": "#ffffff",
        "surface2": "#f9fafb",
        "border": "#e5e7eb",
        "text": "#1f2937",
        "text-muted": "#6b7280",
        "primary": "#667eea",
        "primary2": "#764ba2",
        "accent": "#ec4899",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "shadow": "rgba(0,0,0,0.08)",
        "header-grad": "linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)",
    }

THEME = get_theme_vars(st.session_state.theme)
T = THEME  # shorthand

# ═══════════════════════════════════════════════════════════
#   🎨 Custom CSS
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {{
        --bg: {T['bg']};
        --bg2: {T['bg2']};
        --surface: {T['surface']};
        --surface2: {T['surface2']};
        --border: {T['border']};
        --text: {T['text']};
        --text-muted: {T['text-muted']};
        --primary: {T['primary']};
        --primary2: {T['primary2']};
        --accent: {T['accent']};
        --success: {T['success']};
        --warning: {T['warning']};
        --danger: {T['danger']};
        --shadow: {T['shadow']};
    }}

    html, body, [class*="css"] {{
        font-family: 'Vazirmatn', 'Inter', 'Tahoma', sans-serif;
        color: var(--text);
    }}

    .main .block-container {{
        direction: rtl;
        text-align: right;
        padding-top: 1rem;
        max-width: 1400px;
    }}

    /* ═══ Animated background ═══ */
    .stApp {{
        background:
            radial-gradient(circle at 10% 0%, {T['primary']}15, transparent 45%),
            radial-gradient(circle at 90% 100%, {T['accent']}15, transparent 45%),
            var(--bg);
        background-attachment: fixed;
    }}

    /* ═══ Header ═══ */
    .main-header {{
        background: {T['header-grad']};
        background-size: 200% 200%;
        animation: gradientShift 12s ease infinite;
        padding: 2rem 2.5rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 50px -12px {T['primary']}80;
        position: relative;
        overflow: hidden;
    }}
    @keyframes gradientShift {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%; right: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
        animation: pulse 8s ease-in-out infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); opacity: 0.5; }}
        50% {{ transform: scale(1.15); opacity: 0.9; }}
    }}
    .main-header h1 {{
        margin: 0; font-size: 2.4rem;
        position: relative; z-index: 1; font-weight: 800;
        letter-spacing: -0.02em;
    }}
    .main-header p {{
        margin-top: 0.6rem; opacity: 0.95;
        font-size: 1.05rem; position: relative; z-index: 1;
    }}
    .main-header .badge {{
        display: inline-block;
        background: rgba(255,255,255,0.22);
        padding: 0.35rem 0.95rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        margin-left: 0.4rem;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.25);
        font-weight: 600;
    }}

    /* ═══ Glass surface ═══ */
    .glass {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px -4px var(--shadow);
        backdrop-filter: blur(10px);
    }}

    /* ═══ Source cards ═══ */
    .source-card {{
        background: linear-gradient(135deg, var(--surface2) 0%, var(--surface) 100%);
        border-right: 4px solid var(--primary);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        box-shadow: 0 2px 12px var(--shadow);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    .source-card::after {{
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, var(--primary)08, transparent);
        transform: translateX(100%);
        transition: transform 0.6s;
    }}
    .source-card:hover::after {{
        transform: translateX(-100%);
    }}
    .source-card:hover {{
        transform: translateX(-6px);
        box-shadow: 0 8px 28px -6px {T['primary']}40;
    }}
    .source-card .page-badge {{
        display: inline-block;
        background: linear-gradient(135deg, var(--primary), var(--primary2));
        color: white;
        padding: 0.25rem 0.85rem;
        border-radius: 12px;
        font-size: 0.78rem;
        margin-bottom: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    .source-card .score-badge {{
        display: inline-block;
        background: var(--success);
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 10px;
        font-size: 0.72rem;
        margin-right: 0.4rem;
        font-weight: 600;
    }}
    .source-card p {{
        margin: 0.5rem 0 0 0;
        line-height: 1.9;
        color: var(--text);
        font-size: 0.95rem;
    }}

    /* ═══ Stat boxes ═══ */
    .stat-box {{
        background: var(--surface);
        padding: 1.4rem 1rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px -4px var(--shadow);
        border: 1px solid var(--border);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    .stat-box::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        transform: scaleX(0);
        transform-origin: right;
        transition: transform 0.4s;
    }}
    .stat-box:hover::before {{
        transform: scaleX(1);
    }}
    .stat-box:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 32px -8px {T['primary']}40;
    }}
    .stat-box .stat-icon {{ font-size: 2.6rem; margin-bottom: 0.4rem; }}
    .stat-box .stat-value {{
        font-size: 1.8rem; font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .stat-box .stat-label {{
        color: var(--text-muted); font-size: 0.88rem; margin-top: 0.3rem;
        font-weight: 500;
    }}

    /* ═══ Success badge ═══ */
    .success-badge {{
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 0.9rem 1.5rem;
        border-radius: 14px; text-align: center;
        font-weight: 600;
        box-shadow: 0 6px 20px -6px rgba(17, 153, 142, 0.5);
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }}
    .success-badge .sep {{ opacity: 0.6; }}

    /* ═══ Confidence badges ═══ */
    .conf-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.95rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0.4rem 0;
        letter-spacing: 0.01em;
    }}
    .conf-high {{
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.35);
    }}
    .conf-mid {{
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.35);
    }}
    .conf-low {{
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.35);
    }}

    /* ═══ Chat messages ═══ */
    .stChatMessage {{
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px var(--shadow);
        border: 1px solid var(--border);
    }}

    /* ═══ Suggested chips ═══ */
    .chip {{
        display: inline-block;
        background: linear-gradient(135deg, var(--primary)18, var(--accent)18);
        color: var(--primary);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid var(--primary)30;
        font-weight: 500;
    }}
    .chip:hover {{
        background: linear-gradient(135deg, var(--primary)30, var(--accent)30);
        transform: translateY(-2px);
    }}

    /* ═══ Pinned answer ═══ */
    .pinned-answer {{
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-right: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: #78350f;
    }}

    /* ═══ Hide Streamlit defaults ═══ */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* ═══ Buttons ═══ */
    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
        border: 1px solid var(--border);
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px -2px {T['primary']}50;
        border-color: var(--primary);
    }}

    /* ═══ Metrics ═══ */
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        font-weight: 700;
    }}

    /* ═══ Feature cards ═══ */
    .feature-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.3rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px -4px var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }}
    .feature-card::before {{
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 80px; height: 80px;
        background: radial-gradient(circle, var(--primary)20, transparent 70%);
        border-radius: 50%;
        transform: translate(30%, -30%);
    }}
    .feature-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 16px 40px -10px {T['primary']}50;
        border-color: var(--primary)60;
    }}
    .feature-card .fc-icon {{ font-size: 2.2rem; margin-bottom: 0.6rem; }}
    .feature-card .fc-title {{
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.4rem;
        font-size: 1.05rem;
    }}
    .feature-card .fc-desc {{
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.65;
    }}

    /* ═══ Progress ring ═══ */
    .progress-ring {{
        display: inline-block;
        transform: rotate(-90deg);
    }}

    /* ═══ Scrollbar ═══ */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg2); }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(var(--primary), var(--primary2));
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

    /* ═══ Toast animation ═══ */
    @keyframes slideIn {{
        from {{ transform: translateX(100%); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}

    /* ═══ Footer ═══ */
    .footer {{
        text-align: center;
        padding: 1.5rem 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
    }}
    .footer .name {{
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* ═══ Sidebar tweaks ═══ */
    section[data-testid="stSidebar"] {{
        background: var(--surface);
        border-left: 1px solid var(--border);
    }}

    /* ═══ Tabs ═══ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: var(--surface2);
        padding: 4px;
        border-radius: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#   🔐 Helpers
# ═══════════════════════════════════════════════════════════
def safe_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text), quote=True)


def short_hash(text: str) -> str:
    """Short stable hash for keys."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def format_bytes(num_bytes: int) -> str:
    """Human-readable byte size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def conf_label(confidence: Optional[float]) -> Tuple[str, str]:
    """Return (css_class, label) for a confidence score."""
    if confidence is None:
        return "", ""
    if confidence > 0.75:
        return "conf-high", f"🎯 High confidence ({confidence*100:.0f}%)"
    if confidence > 0.45:
        return "conf-mid", f"⚠️ Medium confidence ({confidence*100:.0f}%)"
    return "conf-low", f"🚨 Low confidence ({confidence*100:.0f}%)"


# ═══════════════════════════════════════════════════════════
#   💾 Cached document processing
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def process_document(
    file_path: str,
    file_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
):
    """Process a document (cached per unique inputs)."""
    vectorstore, stats = build_vectorstore(
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    retriever = get_retriever(vectorstore, k=3)
    return vectorstore, retriever, stats


@st.cache_resource(show_spinner=False)
def get_cached_retriever(
    doc_name: str,
    k: int,
    search_type: str,
    _vectorstore,
):
    """Cache retriever objects per (doc, k, search_type)."""
    return get_retriever(_vectorstore, k=k, search_type=search_type)


# ═══════════════════════════════════════════════════════════
#   💡 Suggested questions from document content
# ═══════════════════════════════════════════════════════════
def generate_suggested_questions(vectorstore, n: int = 4) -> List[str]:
    """Extract simple questions from the first chunks."""
    try:
        data = vectorstore._collection.get(limit=8)
        docs = data.get("documents", [])
        suggestions: List[str] = []
        seen = set()
        for doc in docs[: n * 2]:
            text = doc.strip()
            sentences = re.split(r"[.؟!?\n]", text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 25 and sentence not in seen:
                    seen.add(sentence)
                    suggestions.append(
                        f"What does the document say about: {sentence[:70]}?"
                    )
                    break
            if len(suggestions) >= n:
                break
        return suggestions[:n]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════
#   🎬 Onboarding tour
# ═══════════════════════════════════════════════════════════
if not st.session_state.onboarding_seen:
    with st.expander("👋 Welcome — quick tour (click to expand)", expanded=True):
        st.markdown("""
        ### 🎯 How to use this app

        1. **Upload** one or more PDF files below
        2. **Wait** for processing (10–30 seconds per document)
        3. **Ask** questions in Persian, English, or Arabic
        4. **Get** answers with sources and confidence scores

        ### ⚙️ Sidebar controls
        | Control | Purpose |
        |---|---|
        | **Model** | GPT-4o-mini (fast) or GPT-4o (best) |
        | **Persona** | Assistant / Expert / Tutor / Journalist / Casual |
        | **Style** | Default / Concise / Detailed / Bullets / Academic |
        | **Multi-query** | Ask multiple variations for better recall |
        | **Streaming** | Real-time typewriter effect |
        | **Citations** | Add reference list at the end |

        ### 💡 Pro tips
        - Lower `temperature` (0.1) → more factual
        - Higher `k` → more sources (slower)
        - Use **MMR** for diverse perspectives
        - Pin important answers 📌 to keep them handy
        """)
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("✅ Got it", use_container_width=True):
                st.session_state.onboarding_seen = True
                st.rerun()
        with c2:
            if st.button("Skip tour", use_container_width=True):
                st.session_state.onboarding_seen = True
                st.rerun()

# ═══════════════════════════════════════════════════════════
#   🚀 Header
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
    <h1>🌐 Nexus RAG — Multilingual Intelligence</h1>
    <p>Upload PDFs and ask questions in Persian, English, or Arabic — powered by GPT-4 & ChromaDB</p>
    <span class="badge">⚡ v4.0</span>
    <span class="badge">🧠 {len(AVAILABLE_MODELS)} models</span>
    <span class="badge">🌍 3 languages</span>
    <span class="badge">🎭 {len(PERSONAS)} personas</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#   📊 Sidebar
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    # ─── Brand ───
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 2.4rem;">🌐</div>
        <div style="font-weight: 800; font-size: 1.15rem; 
                    background: linear-gradient(135deg, var(--primary), var(--accent));
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Nexus RAG
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">v4.0 · Pro Edition</div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Theme toggle ───
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        theme_choice = st.radio(
            "Theme",
            options=["☀️ Light", "🌙 Dark"],
            index=0 if st.session_state.theme == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
    with tcol2:
        if st.button("🔄", help="Refresh app state"):
            st.rerun()

    new_theme = "light" if "Light" in theme_choice else "dark"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()

    # ─── Sidebar tabs ───
    tab_chat, tab_docs, tab_stats, tab_danger = st.tabs(
        ["💬 Chat", "📁 Docs", "📊 Stats", "⚠️ Danger"]
    )

    # ═══════════════════════════════════════════
    #   TAB: Chat settings
    # ═══════════════════════════════════════════
    with tab_chat:
        st.markdown("##### 🤖 Model")
        model_options = list(AVAILABLE_MODELS.keys())
        model_labels = [f"{m} — {AVAILABLE_MODELS[m]['desc']}" for m in model_options]
        selected_model_label = st.selectbox(
            "Language Model", model_labels, index=0, label_visibility="collapsed"
        )
        selected_model = model_options[model_labels.index(selected_model_label)]

        temperature = st.slider(
            "🌡️ Temperature", 0.0, 1.0, 0.1, 0.05,
            help="Lower = factual, higher = creative",
        )

        top_k = st.slider(
            "📚 Sources (k)", 1, 10, 3, 1,
            help="How many chunks to retrieve",
        )

        search_type = st.selectbox(
            "🔎 Search Mode",
            ["similarity", "mmr", "similarity_score_threshold"],
            index=0,
            help="MMR = diverse results · threshold = min similarity",
        )

        st.markdown("##### 🎭 Personality")
        persona_labels = [p.capitalize() for p in PERSONAS.keys()]
        selected_persona_label = st.selectbox(
            "Persona", persona_labels, index=0, label_visibility="collapsed"
        )
        selected_persona = list(PERSONAS.keys())[
            persona_labels.index(selected_persona_label)
        ]

        style_labels = list(STYLE_MODIFIERS.keys())
        selected_style = st.selectbox(
            "Response Style", style_labels, index=0, label_visibility="collapsed"
        )

        st.markdown("##### 🎛️ Behavior")
        streaming = st.toggle("⚡ Streaming", value=False)
        use_multiquery = st.toggle("🔀 Multi-query", value=False)
        add_citations = st.toggle("📎 Citations", value=False)
        reformulate = st.toggle("🧠 Reformulate", value=False)
        st.session_state.auto_scroll = st.toggle(
            "⬇️ Auto-scroll", value=st.session_state.auto_scroll
        )

        st.markdown("##### 🧠 Embedding")
        embed_options = list(EMBEDDING_MODELS.keys())
        embed_labels = [
            f"{m.split('/')[-1]} — {EMBEDDING_MODELS[m]['desc']}"
            for m in embed_options
        ]
        selected_embed_label = st.selectbox(
            "Embedding", embed_labels, index=0, label_visibility="collapsed"
        )
        selected_embed_model = embed_options[embed_labels.index(selected_embed_label)]

        with st.expander("🔧 Chunking"):
            chunk_size = st.slider("Chunk size", 200, 2000, 500, 50)
            chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, 10)

    # ═══════════════════════════════════════════
    #   TAB: Documents
    # ═══════════════════════════════════════════
    with tab_docs:
        if st.session_state.documents:
            for doc_name in list(st.session_state.documents.keys()):
                is_active = doc_name == st.session_state.active_doc
                cols = st.columns([5, 1])
                with cols[0]:
                    icon = "🟢" if is_active else "⚪"
                    label = f"{icon} {doc_name[:22]}"
                    if st.button(
                        label,
                        key=f"select_{doc_name}",
                        use_container_width=True,
                    ):
                        st.session_state.active_doc = doc_name
                        st.session_state.messages = []
                        st.rerun()
                with cols[1]:
                    if st.button("🗑️", key=f"del_{doc_name}", help="Remove"):
                        del st.session_state.documents[doc_name]
                        if st.session_state.active_doc == doc_name:
                            st.session_state.active_doc = next(
                                iter(st.session_state.documents), None
                            )
                            st.session_state.messages = []
                        st.rerun()

            if len(st.session_state.documents) > 1:
                if st.button("🗑️ Clear all documents", use_container_width=True):
                    st.session_state.documents = {}
                    st.session_state.active_doc = None
                    st.session_state.messages = []
                    st.rerun()
        else:
            st.caption("📭 No documents loaded")

        st.divider()
        st.markdown("##### 📥 Export chat")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with c2:
            if st.session_state.messages:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "document": st.session_state.active_doc,
                    "model": selected_model,
                    "persona": selected_persona,
                    "style": selected_style,
                    "messages": st.session_state.messages,
                }
                st.download_button(
                    "📥 JSON",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        if st.session_state.messages:
            # Markdown export
            md_lines = [f"# Chat Export — {datetime.now():%Y-%m-%d %H:%M}\n"]
            md_lines.append(f"**Model:** {selected_model}  ")
            md_lines.append(f"**Persona:** {selected_persona}  ")
            md_lines.append(f"**Style:** {selected_style}\n")
            md_lines.append("---\n")
            for m in st.session_state.messages:
                role = "🧑 User" if m["role"] == "user" else "🤖 Assistant"
                md_lines.append(f"## {role}\n\n{m['content']}\n")
                if m.get("sources"):
                    md_lines.append("### 📚 Sources\n")
                    for i, s in enumerate(m["sources"], 1):
                        md_lines.append(
                            f"- **[Source {i}]** Page {s['page']}: "
                            f"{s['content'][:200]}...\n"
                        )

            md_text = "\n".join(md_lines)
            c3, c4 = st.columns(2)
            with c3:
                st.download_button(
                    "📄 Markdown",
                    data=md_text.encode("utf-8"),
                    file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with c4:
                # Plain text export
                txt_lines = []
                for m in st.session_state.messages:
                    role = "USER" if m["role"] == "user" else "ASSISTANT"
                    txt_lines.append(f"[{role}]\n{m['content']}\n")
                    txt_lines.append("-" * 60 + "\n")
                st.download_button(
                    "📝 Text",
                    data="\n".join(txt_lines).encode("utf-8"),
                    file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        # ─── Search history ───
        if st.session_state.search_history:
            with st.expander(
                f"🔎 Search History ({len(st.session_state.search_history)})"
            ):
                for q in reversed(st.session_state.search_history[-12:]):
                    if st.button(
                        f"• {q[:50]}",
                        key=f"hist_{short_hash(q)}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_chip_question = q
                        st.rerun()

    # ═══════════════════════════════════════════
    #   TAB: Stats
    # ═══════════════════════════════════════════
    with tab_stats:
        active = st.session_state.active_doc
        if active and active in st.session_state.documents:
            stats = st.session_state.documents[active].get("stats", {})
            c1, c2 = st.columns(2)
            with c1:
                st.metric("📦 Chunks", stats.get("num_chunks", "—"))
                st.metric(
                    "📄 Pages",
                    stats.get("num_documents", stats.get("num_pages", "—")),
                )
            with c2:
                st.metric(
                    "✍️ Chars",
                    f"{stats.get('total_characters', 0):,}",
                )
                st.metric("⏱️ Time", f"{stats.get('total_time', 0):.1f}s")

        st.divider()
        st.metric("❓ Questions asked", st.session_state.total_questions)

        usage = get_token_usage()
        cost = estimate_cost(
            selected_model,
            usage.get("total_input", 0),
            usage.get("total_output", 0),
        )
        st.metric("💰 API cost (est.)", f"${cost:.4f}")
        st.caption(
            f"📥 Input: {usage.get('total_input', 0):,} · "
            f"📤 Output: {usage.get('total_output', 0):,}"
        )

        disk = get_disk_usage()
        st.caption(f"💾 Disk usage: {disk.get('size_mb', 0):.1f} MB")

        st.divider()
        st.toggle("🐛 Debug mode", key="show_debug")

    # ═══════════════════════════════════════════
    #   TAB: Danger zone
    # ═══════════════════════════════════════════
    with tab_danger:
        st.warning("⚠️ These actions are irreversible!")

        if st.button("🧹 Clear response cache", use_container_width=True):
            clear_response_cache()
            st.toast("Response cache cleared", icon="🧹")

        if st.button("📊 Reset token counters", use_container_width=True):
            reset_token_usage()
            st.toast("Token counters reset", icon="📊")

        if st.button("💾 Backup Vector DB", use_container_width=True):
            try:
                path = backup_vectorstore()
                st.toast(f"Backup saved: {path}", icon="💾")
            except Exception as e:
                st.error(f"Backup failed: {e}")

        if st.button("🗑️ Delete ALL vector data", use_container_width=True):
            delete_all_data()
            st.session_state.documents = {}
            st.session_state.active_doc = None
            st.session_state.messages = []
            st.toast("Vector DB cleared", icon="🗑️")
            st.rerun()

    st.divider()
  

# ═══════════════════════════════════════════════════════════
#   📁 File uploader
# ═══════════════════════════════════════════════════════════
uploaded_files = st.file_uploader(
    "📄 Upload PDF document(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Drag and drop one or more PDFs (max ~50 MB each)",
)

# ═══════════════════════════════════════════════════════════
#   🗂️ Process uploaded files
# ═══════════════════════════════════════════════════════════
if uploaded_files:
    os.makedirs("docs", exist_ok=True)

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.documents:
            continue

        temp_path = os.path.join("docs", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_hash = f"{uploaded_file.name}_{uploaded_file.size}"
        file_mb = uploaded_file.size / 1024 / 1024
        
        if file_mb > 5:
            st.warning(
                f"⚠️ «{uploaded_file.name}» is **{file_mb:.1f} MB** — "
                f"processing may take **2-5 minutes** for embedding. Please wait..."
            )

        progress = st.progress(0, text=f"🔄 Processing «{uploaded_file.name}» ({file_mb:.1f} MB)...")

        try:
            progress.progress(10, text="📄 Saving file...")
            time.sleep(0.05)

            progress.progress(30, text=f"✂️ Chunking & embedding (may take a while)...")
            
            t0 = time.time()
            vectorstore, retriever, stats = process_document(
                temp_path,
                file_hash,
                chunk_size,
                chunk_overlap,
                selected_embed_model,
            )
            elapsed = time.time() - t0

            progress.progress(90, text=f"💾 Saving vectors ({elapsed:.1f}s)...")

            # 🔧 SAFE stats extraction — works with ANY key naming
            safe_stats = {
                "num_chunks": (
                    stats.get("num_chunks")
                    or stats.get("chunk_count")
                    or stats.get("chunks")
                    or len(vectorstore._collection.get().get("ids", []))
                ),
                "num_documents": (
                    stats.get("num_documents")
                    or stats.get("num_pages")
                    or stats.get("pages")
                    or stats.get("n_pages")
                    or "?"
                ),
                "total_characters": (
                    stats.get("total_characters")
                    or stats.get("total_chars")
                    or stats.get("chars")
                    or 0
                ),
                "total_time": stats.get("total_time") or elapsed,
            }

            st.session_state.documents[uploaded_file.name] = {
                "vectorstore": vectorstore,
                "retriever": retriever,
                "stats": safe_stats,  # ← safe version
                "path": temp_path,
                "embedding_model": selected_embed_model,
            }

            if st.session_state.active_doc is None:
                st.session_state.active_doc = uploaded_file.name
                st.session_state.messages = []

            progress.progress(100, text="✅ Done!")
            time.sleep(0.3)
            progress.empty()

            st.success(
                f"✅ «{uploaded_file.name}» ready — "
                f"{safe_stats['num_chunks']} chunks in {elapsed:.1f}s"
            )

        except Exception as e:
            progress.empty()
            st.error(f"❌ Failed to process «{uploaded_file.name}»: {e}")
            with st.expander("🔍 Full error traceback", expanded=True):
                import traceback
                st.code(traceback.format_exc(), language="python")
            st.info(
                "**Common causes:**\n"
                "1. **OpenAI API key missing/invalid** — check `.env`\n"
                "2. **Rate limit reached** — wait 60s and retry\n"
                "3. **File too large** — try a PDF under 5 MB\n"
                "4. **PDF has no extractable text** — scanned image PDFs need OCR\n"
                "5. **Internet connection issue**\n"
            )
            continue
# ═══════════════════════════════════════════════════════════
#   💬 Chat area
# ═══════════════════════════════════════════════════════════
active = st.session_state.active_doc

if active and active in st.session_state.documents:
    doc_data = st.session_state.documents[active]
    vectorstore = doc_data["vectorstore"]
    stats = doc_data.get("stats", {})

    st.markdown(f"""
    <div class="success-badge">
        <span>✅ Active: «{safe_html(active)}»</span>
        <span class="sep">·</span>
        <span>{stats.get('num_chunks', '?')} chunks</span>
        <span class="sep">·</span>
        <span>{stats.get('num_documents', stats.get('num_pages', '?'))} pages</span>
        <span class="sep">·</span>
        <span>⏱️ {stats.get('total_time', 0):.1f}s</span>
    </div>
    """, unsafe_allow_html=True)

    # ─── Pinned answers ───
    if st.session_state.pinned_answers:
        with st.expander(
            f"📌 Pinned Answers ({len(st.session_state.pinned_answers)})",
            expanded=False,
        ):
            for i, pin in enumerate(st.session_state.pinned_answers, 1):
                st.markdown(f"""
                <div class="pinned-answer">
                    <strong>Q{i}:</strong> {safe_html(pin['question'])}<br><br>
                    <strong>A:</strong> {safe_html(pin['answer'][:400])}...
                </div>
                """, unsafe_allow_html=True)
            if st.button("🗑️ Clear all pins", use_container_width=True):
                st.session_state.pinned_answers = []
                st.rerun()

    st.divider()

    # ═══════════════════════════════════════════
    #   Render chat history
    # ═══════════════════════════════════════════
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Confidence badge
            if msg.get("confidence") is not None:
                cls, label = conf_label(msg["confidence"])
                if cls:
                    st.markdown(
                        f'<span class="conf-badge {cls}">{label}</span>',
                        unsafe_allow_html=True,
                    )

            # Sources
            if msg.get("sources"):
                with st.expander(f"📌 {len(msg['sources'])} source(s)"):
                    for i, src in enumerate(msg["sources"], 1):
                        score_html = ""
                        if "score" in src and src["score"] is not None:
                            score_html = (
                                f'<span class="score-badge">'
                                f'score: {src["score"]:.2f}</span>'
                            )
                        content = safe_html(src["content"][:500])
                        ellipsis = "..." if len(src["content"]) > 500 else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="page-badge">Source {i} — Page {src.get('page', '?')}</span>
                            {score_html}
                            <p>{content}{ellipsis}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    #   Suggested questions (empty chat)
    # ═══════════════════════════════════════════
    if not st.session_state.messages:
        suggestions = generate_suggested_questions(vectorstore, n=4)
        if suggestions:
            st.markdown("### 💡 Try asking:")
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(
                        f"💬 {q[:70]}",
                        key=f"suggest_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_chip_question = q
                        st.rerun()

    st.divider()

    # ═══════════════════════════════════════════
    #   Chat input & processing
    # ═══════════════════════════════════════════
    question = st.chat_input("Ask your question in Persian, English, or Arabic...")

    if st.session_state.pending_chip_question and not question:
        question = st.session_state.pending_chip_question
        st.session_state.pending_chip_question = None

    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.search_history.append(question)

        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            try:
                # Optional reformulation
                effective_question = question
                if reformulate:
                    with st.spinner("🧠 Reformulating question..."):
                        better = reformulate_question(question, model=selected_model)
                        if better and better != question:
                            effective_question = better
                            st.caption(f"🔍 Searching for: _{better}_")

                # Get retriever
                retriever = get_cached_retriever(
                    active, top_k, search_type, vectorstore
                )

                sources: List[Dict[str, Any]] = []
                confidence: Optional[float] = None
                answer: str = ""

                # ─── Streaming mode ───
                if streaming:
                    placeholder = st.empty()
                    full_answer = ""
                    for chunk in ask_question_stream(
                        retriever,
                        effective_question,
                        model=selected_model,
                        temperature=temperature,
                        style=selected_style,
                        persona=selected_persona,
                    ):
                        full_answer += chunk
                        placeholder.markdown(full_answer + "▌")
                    placeholder.markdown(full_answer)
                    answer = full_answer

                # ─── Standard mode ───
                else:
                    with st.spinner("🔍 Searching document..."):
                        answer, sources, confidence = ask_question(
                            retriever,
                            effective_question,
                            model=selected_model,
                            temperature=temperature,
                            style=selected_style,
                            persona=selected_persona,
                            add_citations=add_citations,
                            return_confidence=True,
                        )

                    st.markdown(answer)

                    if confidence is not None:
                        cls, label = conf_label(confidence)
                        if cls:
                            st.markdown(
                                f'<span class="conf-badge {cls}">{label}</span>',
                                unsafe_allow_html=True,
                            )

                    if sources:
                        with st.expander(f"📌 {len(sources)} source(s)"):
                            for i, src in enumerate(sources, 1):
                                score_html = ""
                                if "score" in src and src["score"] is not None:
                                    score_html = (
                                        f'<span class="score-badge">'
                                        f'score: {src["score"]:.2f}</span>'
                                    )
                                content = safe_html(src["content"][:500])
                                ellipsis = "..." if len(src["content"]) > 500 else ""
                                st.markdown(f"""
                                <div class="source-card">
                                    <span class="page-badge">Source {i} — Page {src.get('page', '?')}</span>
                                    {score_html}
                                    <p>{content}{ellipsis}</p>
                                </div>
                                """, unsafe_allow_html=True)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })
                st.session_state.total_questions += 1
                st.session_state.last_question = question

                # ─── Message actions ───
                col_a, col_b, col_c, _ = st.columns([1, 1, 1, 3])
                msg_id = len(st.session_state.messages)

                with col_a:
                    if st.button("📌 Pin", key=f"pin_{msg_id}"):
                        st.session_state.pinned_answers.append({
                            "question": question,
                            "answer": answer,
                        })
                        st.toast("Pinned!", icon="📌")

                with col_b:
                    if st.button("🔄 Retry", key=f"retry_{msg_id}"):
                        # Remove last 2 messages (user + assistant)
                        st.session_state.messages = st.session_state.messages[:-2]
                        st.session_state.pending_chip_question = question
                        st.rerun()

                with col_c:
                    if st.button("🗑️ Delete", key=f"delmsg_{msg_id}"):
                        st.session_state.messages = st.session_state.messages[:-2]
                        st.rerun()

                # Debug panel
                if st.session_state.get("show_debug"):
                    with st.expander("🐛 Debug info"):
                        st.json({
                            "question": question,
                            "effective_question": effective_question,
                            "model": selected_model,
                            "temperature": temperature,
                            "top_k": top_k,
                            "search_type": search_type,
                            "persona": selected_persona,
                            "style": selected_style,
                            "sources_count": len(sources),
                            "confidence": confidence,
                        })

            except Exception as e:
                st.error(f"❌ Error: {e}")
                with st.expander("🔍 Full traceback"):
                    st.exception(e)

# ═══════════════════════════════════════════════════════════
#   👋 Welcome screen (no active document)
# ═══════════════════════════════════════════════════════════
else:
    st.info("👆 Upload a PDF document to get started")

    # ─── Stats strip ───
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-icon">📚</div>
            <div class="stat-value">3</div>
            <div class="stat-label">Languages</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-icon">🧠</div>
            <div class="stat-value">6+</div>
            <div class="stat-label">Embedding models</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-icon">🎭</div>
            <div class="stat-value">25</div>
            <div class="stat-label">Prompt variations</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">0.8s</div>
            <div class="stat-label">Avg response</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ✨ Features")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="fc-icon">🌐</div>
            <div class="fc-title">Multilingual</div>
            <div class="fc-desc">
                Auto-detects Persian, English, and Arabic.
                Answers in the same language as your question.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <div class="fc-icon">📁</div>
            <div class="fc-title">Multi-Document</div>
            <div class="fc-desc">
                Upload multiple PDFs. Switch between them
                instantly from the sidebar.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="fc-icon">🎭</div>
            <div class="fc-title">Personas & Styles</div>
            <div class="fc-desc">
                5 personas × 5 styles = 25 prompt variations.
                Tutor, expert, journalist, casual, and more.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <div class="fc-icon">🎯</div>
            <div class="fc-title">Confidence Scoring</div>
            <div class="fc-desc">
                Every answer gets a confidence badge based on
                source quality and count.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="fc-icon">💰</div>
            <div class="fc-title">Cost Tracking</div>
            <div class="fc-desc">
                Live USD cost estimation based on token usage.
                See exactly what each question costs.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <div class="fc-icon">📊</div>
            <div class="fc-title">Export & Backup</div>
            <div class="fc-desc">
                Download chats as JSON, Markdown, or plain text.
                Backup the entire vector DB with one click.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#   🦶 Footer
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
    <div class="footer">
        Built with ❤️ by <span class="name">A_srt343</span><br>
        Instagram: <span class="name">@project_srt343</span><br>
        <span style="font-size: 0.8rem; opacity: 0.7;">
            Nexus RAG v4.0 · Powered by Streamlit, LangChain, ChromaDB & GPT
        </span>
    </div>
""", unsafe_allow_html=True)