"""
ui_components.py — Takshashila KnowledgeBase RAG
Palette: takshashila.org.in — maroon #7B1D3A, gold #C9922A, white page bg
"""

from typing import Dict, List
import streamlit as st
from src.utils import truncate, clean_mojibake_text, clean_url_value

# ── Design tokens ──────────────────────────────────────────────────────────────
MAROON   = "#7B1D3A"
MAROON_D = "#5A1229"
MAROON_M = "#9B2A4D"   # medium maroon — lighter, friendlier hero/accent fill
MAROON_L = "#F9EFF3"
GOLD     = "#C9922A"
GOLD_L   = "#FEF7ED"
WHITE    = "#FFFFFF"
OFF_WHT  = "#F8F4F0"
BORDER   = "#E2D5C8"
TEXT     = "#1A0E06"
MUTED    = "#6B5347"
BLUE     = "#1A4F8C"
GREEN    = "#1B6640"
PURPLE   = "#4A1C6F"

# Hero gradient stops — medium maroon → deep maroon (no near-black end),
# always paired with light text for strong contrast.
HERO_FROM = MAROON_M
HERO_TO   = MAROON_D

# Colour per source key (used for source cards / pills).
SOURCE_COLOURS = {
    "commit_kb":      MAROON,
    "staff_handbook": "#7A5000",
    "publication":    BLUE,
    "blog":           GREEN,
    "pdf":            PURPLE,
    "local":          MUTED,
}

# Short uppercase label per source key.
SOURCE_LABELS = {
    "commit_kb":      "COMMIT KB",
    "staff_handbook": "HANDBOOK",
    "publication":    "PUBLICATION",
    "blog":           "BLOG",
    "pdf":            "PDF",
    "local":          "LOCAL",
}


def _source_colour(source: str) -> str:
    return SOURCE_COLOURS.get((source or "").lower(), MUTED)


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get((source or "").lower(), (source or "DOC").upper())


def confidence_badge(level: str) -> str:
    c = {"high": GREEN, "medium": "#7A5000", "low": "#8B1A1A", "none": "#555"}.get(level, "#555")
    return (
        f'<span style="background:{c};color:#fff;padding:3px 14px;'
        f'border-radius:20px;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.08em;font-family:Inter,sans-serif">{level.upper()}</span>'
    )


def low_confidence_warning(level: str):
    """Render a soft warning when retrieval confidence is low/none."""
    if level == "low":
        st.warning(
            "⚠️ **Low confidence.** The retrieved passages are only weakly "
            "related to your question — treat this answer with caution and "
            "verify against the cited sources."
        )
    elif level == "none":
        st.info(
            "ℹ️ No sufficiently relevant evidence was found in the knowledge "
            "base for this question."
        )


def source_card(chunk: Dict, index: int):
    title    = clean_mojibake_text(chunk.get("title", "Untitled")) or "Untitled"
    source   = chunk.get("source", chunk.get("source_type", "")) or ""
    src_name = clean_mojibake_text(chunk.get("source_name", "") or source)
    category = clean_mojibake_text(chunk.get("category", "") or "")
    url      = clean_url_value(chunk.get("url") or chunk.get("original_url") or "")
    date     = chunk.get("date", "")
    page     = chunk.get("page_number")
    # Prefer raw cosine score for the "match" badge; fall back to fused score.
    score    = chunk.get("score", chunk.get("rrf_score", 0)) or 0

    page_str  = f" · Page {page}" if page else ""
    date_str  = f" &nbsp;·&nbsp; 📅 {date}" if date else ""
    score_pct = f"{score*100:.1f}%" if score else "—"
    tc        = _source_colour(source)
    label     = _source_label(source)
    cat_chip  = ""
    if category:
        cat_chip = (f"<span style='background:{GOLD_L};color:#7A5000;font-size:0.62rem;"
                    f"font-weight:700;padding:2px 9px;border-radius:3px;letter-spacing:.05em;"
                    f"font-family:Inter,sans-serif;text-transform:uppercase'>{category}</span>")

    link_html = ""
    if url:
        link_html = (f"<a href='{url}' target='_blank' style='color:{tc};"
                     f"font-size:0.77rem;font-weight:600;text-decoration:none;"
                     f"font-family:Inter,sans-serif;border:1.5px solid {tc};"
                     f"padding:3px 11px;border-radius:5px;transition:all .15s;"
                     f"display:inline-block'>🔗 Open page</a>")

    st.markdown(
        f"""<div style="background:{WHITE};border:1px solid {BORDER};
                border-left:4px solid {tc};border-radius:10px;
                padding:16px 20px;margin-bottom:10px;
                box-shadow:0 2px 8px rgba(0,0,0,.05);
                transition:box-shadow .2s,transform .15s"
             onmouseover="this.style.boxShadow='0 8px 24px rgba(0,0,0,.12)';this.style.transform='translateY(-2px)'"
             onmouseout="this.style.boxShadow='0 2px 8px rgba(0,0,0,.05)';this.style.transform=''">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
            <div style="flex:1;min-width:0">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;flex-wrap:wrap">
                <span style="background:{tc};color:#fff;font-size:0.62rem;font-weight:700;
                             padding:2px 9px;border-radius:3px;letter-spacing:0.1em;
                             font-family:Inter,sans-serif">{label}</span>
                {cat_chip}
                <span style="color:{MUTED};font-size:0.76rem;font-family:Inter,sans-serif">
                  Source [{index}]{page_str}</span>
              </div>
              <div style="font-weight:700;color:{TEXT};font-size:0.93rem;
                          font-family:Georgia,serif;line-height:1.35;margin-bottom:5px">
                {truncate(title, 110)}</div>
              <div style="color:{MUTED};font-size:0.78rem;font-family:Inter,sans-serif">
                📚 {truncate(src_name, 60)}{date_str}</div>
            </div>
            <div style="text-align:center;min-width:58px;flex-shrink:0">
              <div style="font-size:1.25rem;font-weight:800;color:{tc};
                          font-family:Inter,sans-serif;line-height:1">{score_pct}</div>
              <div style="font-size:0.6rem;color:{MUTED};letter-spacing:.07em;
                          font-family:Inter,sans-serif;text-transform:uppercase">match</div>
            </div>
          </div>
          <div style="margin-top:10px;padding-top:10px;border-top:1px solid {BORDER};
                      font-size:0.81rem;color:{MUTED};font-style:italic;
                      line-height:1.65;font-family:Inter,sans-serif">
            {truncate(clean_mojibake_text(chunk.get("text","")), 280)}</div>
          <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
            {link_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_source_cards(chunks: List[Dict]):
    if not chunks:
        st.info("No sources retrieved.")
        return
    for i, ch in enumerate(chunks, 1):
        source_card(ch, i)


def stat_card(label: str, value, icon: str = "", colour: str = MAROON):
    st.markdown(
        f"""<div style="background:{WHITE};border:1px solid {BORDER};
                border-top:3px solid {colour};border-radius:10px;
                padding:22px 16px 18px;text-align:center;
                box-shadow:0 2px 8px rgba(0,0,0,.05);
                transition:transform .18s,box-shadow .18s;cursor:default"
             onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 10px 28px rgba(0,0,0,.10)'"
             onmouseout="this.style.transform='';this.style.boxShadow='0 2px 8px rgba(0,0,0,.05)'">
          <div style="font-size:0.65rem;color:{MUTED};text-transform:uppercase;
                      letter-spacing:0.14em;font-family:Inter,sans-serif;
                      font-weight:700;margin-bottom:10px">{icon} {label}</div>
          <div style="font-size:2.2rem;font-weight:800;color:{colour};
                      font-family:Georgia,serif;line-height:1">{value:,}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def inject_global_css():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    html,body{{ font-size:16px; }}
    .stApp{{ background:{OFF_WHT} !important; }}
    .main .block-container{{
        padding-top:1.6rem !important;
        padding-bottom:3rem !important;
        max-width:1300px !important;
    }}

    /* ── Sidebar — white, fully readable ── */
    section[data-testid="stSidebar"]{{
        background:{WHITE} !important;
        border-right:1px solid {BORDER} !important;
        box-shadow:3px 0 16px rgba(0,0,0,.07) !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label{{
        color:{TEXT} !important;
    }}

    /* ── All labels ── */
    label, .stSelectbox label, .stTextInput label,
    .stSlider label, .stCheckbox label, .stMultiSelect label{{
        color:{MUTED} !important;
        font-family:Inter,sans-serif !important;
        font-size:0.72rem !important;
        font-weight:700 !important;
        text-transform:uppercase !important;
        letter-spacing:0.1em !important;
    }}

    /* ── Selectbox ── */
    .stSelectbox > div > div{{
        background:{WHITE} !important;
        border:1.5px solid {BORDER} !important;
        border-radius:8px !important;
        color:{TEXT} !important;
        font-family:Inter,sans-serif !important;
        font-size:0.9rem !important;
        transition:border-color .18s,box-shadow .18s !important;
    }}
    .stSelectbox > div > div:hover{{
        border-color:{MAROON} !important;
        box-shadow:0 0 0 3px {MAROON_L} !important;
    }}
    .stSelectbox > div > div:focus-within{{
        border-color:{MAROON} !important;
        box-shadow:0 0 0 3px rgba(123,29,58,.18) !important;
    }}
    .stSelectbox [data-baseweb="select"] span{{
        color:{TEXT} !important;
        font-family:Inter,sans-serif !important;
        font-size:0.9rem !important;
    }}

    /* ── Dropdown popup list ── */
    [data-baseweb="popover"]{{
        background:{WHITE} !important;
        border:1.5px solid {BORDER} !important;
        border-radius:10px !important;
        box-shadow:0 10px 36px rgba(0,0,0,.16) !important;
        overflow:hidden !important;
    }}
    [data-baseweb="menu"] ul{{
        background:{WHITE} !important;
        padding:4px !important;
    }}
    [data-baseweb="menu"] li{{
        color:{TEXT} !important;
        font-family:Inter,sans-serif !important;
        font-size:0.88rem !important;
        border-radius:6px !important;
        padding:8px 12px !important;
        transition:background .12s !important;
    }}
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [aria-selected="true"]{{
        background:{MAROON_L} !important;
        color:{MAROON} !important;
    }}

    /* ── Text inputs ── */
    .stTextInput > div > div > input{{
        background:{WHITE} !important;
        border:1.5px solid {BORDER} !important;
        border-radius:8px !important;
        color:{TEXT} !important;
        font-family:Inter,sans-serif !important;
        font-size:0.9rem !important;
        padding:8px 12px !important;
        transition:border-color .18s,box-shadow .18s !important;
    }}
    .stTextInput > div > div > input:hover{{
        border-color:{MAROON} !important;
    }}
    .stTextInput > div > div > input:focus{{
        border-color:{MAROON} !important;
        box-shadow:0 0 0 3px rgba(123,29,58,.18) !important;
        outline:none !important;
    }}
    .stTextInput > div > div > input::placeholder{{
        color:#B8A090 !important;
    }}

    /* ── Slider ── */
    [data-baseweb="slider"] [role="slider"]{{
        background:{MAROON} !important;
        border:2px solid {WHITE} !important;
        box-shadow:0 0 0 2px {MAROON} !important;
    }}
    [data-baseweb="slider"] [data-testid="stThumbValue"]{{
        color:{MAROON} !important;
        font-family:Inter,sans-serif !important;
        font-weight:700 !important;
        font-size:0.8rem !important;
    }}
    [data-baseweb="slider"] div[data-testid="stSlider"] > div > div > div:nth-child(1) > div:nth-child(2){{
        background:{MAROON} !important;
    }}

    /* ── Checkbox ── */
    [data-testid="stCheckbox"] label span:first-child{{
        border:2px solid {BORDER} !important;
        border-radius:4px !important;
        background:{WHITE} !important;
        transition:all .15s !important;
    }}
    [data-testid="stCheckbox"] input:checked + label span:first-child{{
        background:{MAROON} !important;
        border-color:{MAROON} !important;
    }}

    /* ── Buttons ── */
    .stButton > button{{
        background:{MAROON} !important;
        color:{WHITE} !important;
        border:none !important;
        border-radius:8px !important;
        font-family:Inter,sans-serif !important;
        font-size:0.88rem !important;
        font-weight:600 !important;
        padding:0.55rem 1.25rem !important;
        letter-spacing:0.02em !important;
        transition:background .18s,transform .12s,box-shadow .18s !important;
        box-shadow:0 2px 8px rgba(123,29,58,.2) !important;
    }}
    .stButton > button:hover{{
        background:{MAROON_D} !important;
        transform:translateY(-1px) !important;
        box-shadow:0 6px 18px rgba(123,29,58,.32) !important;
    }}
    .stButton > button:active{{ transform:translateY(0) !important; }}
    .stDownloadButton > button{{
        background:{WHITE} !important;
        color:{MAROON} !important;
        border:1.5px solid {MAROON} !important;
        border-radius:8px !important;
        font-family:Inter,sans-serif !important;
        font-weight:600 !important;
        font-size:0.85rem !important;
        padding:0.45rem 1.1rem !important;
        transition:all .18s !important;
    }}
    .stDownloadButton > button:hover{{
        background:{MAROON} !important;
        color:{WHITE} !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"]{{
        background:{WHITE} !important;
        border-bottom:2px solid {BORDER} !important;
        gap:0 !important;
        padding:0 8px !important;
    }}
    .stTabs [data-baseweb="tab"]{{
        font-family:Inter,sans-serif !important;
        font-size:0.875rem !important;
        font-weight:600 !important;
        color:{MUTED} !important;
        padding:11px 22px !important;
        border-radius:0 !important;
        border-bottom:3px solid transparent !important;
        margin-bottom:-2px !important;
        transition:color .18s,background .18s !important;
        background:transparent !important;
    }}
    .stTabs [data-baseweb="tab"]:hover{{
        color:{MAROON} !important;
        background:{MAROON_L} !important;
    }}
    .stTabs [aria-selected="true"]{{
        color:{MAROON} !important;
        border-bottom:3px solid {MAROON} !important;
        background:transparent !important;
    }}
    .stTabs [data-baseweb="tab-panel"]{{ padding:24px 0 0 !important; }}

    /* ── Chat input ── */
    [data-testid="stChatInput"]{{
        border:2px solid {BORDER} !important;
        border-radius:16px !important;
        background:{WHITE} !important;
        box-shadow:0 4px 20px rgba(0,0,0,.08) !important;
        transition:border-color .2s,box-shadow .22s !important;
        overflow:hidden !important;
        margin-top:8px !important;
    }}
    [data-testid="stChatInput"]:focus-within{{
        border-color:{MAROON} !important;
        box-shadow:0 4px 28px rgba(123,29,58,.22) !important;
        transform:translateY(-1px) !important;
    }}
    [data-testid="stChatInput"] textarea{{
        background:{WHITE} !important;
        color:{TEXT} !important;
        font-family:Inter,sans-serif !important;
        font-size:0.95rem !important;
        border:none !important;
        padding:14px 16px !important;
        line-height:1.55 !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder{{
        color:#B8A090 !important;
        font-style:italic !important;
    }}
    [data-testid="stChatInput"] button{{
        background:{MAROON} !important;
        border-radius:10px !important;
        margin:6px !important;
        transition:background .15s,transform .12s !important;
    }}
    [data-testid="stChatInput"] button:hover{{
        background:{MAROON_D} !important;
        transform:scale(1.05) !important;
    }}
    [data-testid="stChatInput"] button svg{{ fill:{WHITE} !important; }}

    /* ── Chat messages ── */
    [data-testid="stChatMessage"]{{
        background:{WHITE} !important;
        border:1px solid {BORDER} !important;
        border-radius:12px !important;
        margin-bottom:10px !important;
        box-shadow:0 1px 6px rgba(0,0,0,.05) !important;
    }}

    /* ── Expanders ── */
    details{{
        background:{WHITE} !important;
        border:1px solid {BORDER} !important;
        border-radius:10px !important;
        margin-bottom:8px !important;
        overflow:hidden !important;
        transition:box-shadow .15s !important;
    }}
    details:hover{{ box-shadow:0 3px 12px rgba(0,0,0,.08) !important; }}
    details summary{{
        color:{MAROON} !important;
        font-weight:600 !important;
        font-family:Inter,sans-serif !important;
        font-size:0.87rem !important;
        padding:11px 16px !important;
        cursor:pointer !important;
        transition:background .15s !important;
    }}
    details summary:hover{{ background:{MAROON_L} !important; }}
    details[open] summary{{ border-bottom:1px solid {BORDER} !important; }}

    /* ── Alerts ── */
    [data-testid="stAlert"]{{
        border-radius:10px !important;
        border-left-width:4px !important;
        font-family:Inter,sans-serif !important;
    }}

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div > div{{
        background:linear-gradient(90deg,{MAROON},{GOLD}) !important;
        border-radius:4px !important;
    }}

    /* ── Metrics ── */
    [data-testid="metric-container"]{{
        background:{WHITE} !important;
        border:1px solid {BORDER} !important;
        border-radius:10px !important;
        padding:16px 20px !important;
    }}
    [data-testid="stMetricValue"]{{
        color:{MAROON} !important;
        font-family:Georgia,serif !important;
        font-weight:700 !important;
    }}
    [data-testid="stMetricLabel"]{{
        color:{MUTED} !important;
        font-family:Inter,sans-serif !important;
        font-size:0.78rem !important;
    }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"]{{
        border:1px solid {BORDER} !important;
        border-radius:10px !important;
        overflow:hidden !important;
    }}

    /* ── Typography ── */
    h1,h2,h3{{ font-family:'Playfair Display',Georgia,serif !important; color:{TEXT} !important; }}
    h2{{ font-size:1.6rem !important; font-weight:700 !important; }}
    h3{{ font-size:1.15rem !important; font-weight:600 !important; }}
    p,li{{ font-family:Inter,sans-serif !important; color:{TEXT} !important; }}

    /* ── Multiselect tags ── */
    [data-baseweb="tag"]{{
        background:{MAROON} !important;
        color:{WHITE} !important;
        border-radius:5px !important;
    }}

    /* ── Divider ── */
    hr{{ border-color:{BORDER} !important; opacity:1 !important; }}

    /* ── Contrast safety: anything on a maroon fill uses light text ── */
    .tk-hero, .tk-hero *{{ color:rgba(255,255,255,.92) !important; }}
    .tk-hero h1{{ color:#FFFFFF !important; }}
    .tk-hero .tk-eyebrow, .tk-hero strong{{ color:{GOLD} !important; }}
    /* Slightly heavier body text improves on-screen readability of cream cards */
    p,li{{ font-weight:400 !important; }}
    .stMarkdown p{{ line-height:1.6 !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar{{ width:5px; height:5px; }}
    ::-webkit-scrollbar-track{{ background:{OFF_WHT}; }}
    ::-webkit-scrollbar-thumb{{ background:{BORDER}; border-radius:3px; }}
    ::-webkit-scrollbar-thumb:hover{{ background:{MUTED}; }}
    </style>""", unsafe_allow_html=True)