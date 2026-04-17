"""RiskLabs Analyzer — Streamlit UI.

Reproduces the LLM feature-extraction layer of RiskLabs (Cao et al. 2024-2025,
arXiv:2404.07452). Figures 2 (ECC Analyzer) + 3 (News Enrichment Pipeline).

Theme: QuantSeras Design System (Material Dark, #121212, desaturated green #69F0AE).
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

from risklabs_pipeline import (
    DEMO_SCENARIOS,
    analyze_ecc,
    enrich_news,
    interpret_risk_level,
    synthesize_risk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """3-tier key resolution: st.secrets → env var → sidebar input."""
    try:
        val = st.secrets.get("OPENAI_API_KEY", "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


def parse_news_text(raw: str) -> list[str]:
    """Split news textarea text into individual items on blank lines."""
    if not raw.strip():
        return []
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [raw.strip()]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RiskLabs Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# QuantSeras CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

:root {
    --qs-bg: #121212;
    --qs-s1: #1D1D1D;
    --qs-s2: #212121;
    --qs-s3: #242424;
    --qs-s8: #2C2C2C;
    --qs-p: #69F0AE;
    --qs-pv: #00C853;
    --qs-sec: #03DAC6;
    --qs-profit: #00E676;
    --qs-loss: #FF5252;
    --qs-warn: #FFB74D;
    --qs-th: rgba(255,255,255,0.87);
    --qs-tm: rgba(255,255,255,0.60);
    --qs-td: rgba(255,255,255,0.38);
    --qs-b: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', 'IBM Plex Sans', -apple-system, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(105,240,174,0.06) 0%, transparent 45%),
        radial-gradient(circle at 100% 100%, rgba(3,218,198,0.04) 0%, transparent 45%),
        var(--qs-bg);
    color: var(--qs-th);
}

/* Hero */
.qs-hero {
    text-align: center;
    padding: .75rem 0 1rem;
    border-bottom: 1px solid var(--qs-b);
    margin-bottom: 1.25rem;
}
.qs-hero h1 {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #69F0AE, #03DAC6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.qs-hero .tag {
    color: var(--qs-tm);
    font-size: .78rem;
    margin-top: .35rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.qs-hero .ref {
    color: var(--qs-td);
    font-size: .7rem;
    margin-top: .25rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--qs-s1);
    border-right: 1px solid var(--qs-b);
}
.qs-brand {
    text-align: center;
    padding: .5rem 0 .9rem;
    border-bottom: 1px solid var(--qs-b);
    margin-bottom: .75rem;
}
.qs-brand h2 {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #69F0AE, #03DAC6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.qs-brand p {
    color: var(--qs-tm);
    font-size: .7rem;
    margin-top: .2rem;
}
.qs-sec {
    font-size: .66rem;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: var(--qs-tm);
    margin: .9rem 0 .35rem;
    font-weight: 700;
}

/* Risk card */
.qs-risk-card {
    background: var(--qs-s1);
    border: 1px solid var(--qs-b);
    border-radius: 14px;
    padding: 1.75rem 2rem 1.5rem;
    text-align: center;
    margin: .75rem 0 1.25rem;
}
.qs-score {
    font-size: 5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    letter-spacing: -.04em;
    margin: .3rem 0 .1rem;
}
.qs-level {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: .75rem;
}
.qs-row {
    display: flex;
    gap: .5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: .75rem;
}
.qs-chip {
    background: var(--qs-s2);
    border: 1px solid var(--qs-b);
    border-radius: 6px;
    padding: .4rem .7rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    color: var(--qs-tm);
}
.qs-chip strong { color: var(--qs-th); font-weight: 700; margin-left: .25rem; }
.qs-chip.elevated strong { color: var(--qs-loss); }
.qs-chip.moderate strong { color: var(--qs-warn); }
.qs-chip.subdued strong { color: var(--qs-profit); }
.qs-chip.conf-low strong { color: var(--qs-warn); }
.qs-chip.conf-med strong { color: var(--qs-sec); }
.qs-chip.conf-high strong { color: var(--qs-profit); }
.qs-narrative {
    color: var(--qs-th);
    font-size: .87rem;
    line-height: 1.6;
    margin-top: .9rem;
    padding-top: .9rem;
    border-top: 1px solid var(--qs-b);
    text-align: left;
}

/* Misc content cards */
.qs-card {
    background: var(--qs-s1);
    border: 1px solid var(--qs-b);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin: .5rem 0;
}
.qs-card h4 {
    color: var(--qs-p);
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin: 0 0 .6rem;
}
.qs-card p, .qs-card li {
    color: var(--qs-th);
    font-size: .87rem;
    line-height: 1.6;
    margin: 0 0 .3rem;
}
.qs-risk-bullet { color: var(--qs-loss); }
.qs-ok-bullet { color: var(--qs-profit); }

/* Feature intro boxes */
.qs-feat {
    background: var(--qs-s1);
    border: 1px solid var(--qs-b);
    border-radius: 8px;
    padding: 1.1rem;
    height: 100%;
}
.qs-feat .icon { font-size: 1.7rem; margin-bottom: .4rem; }
.qs-feat h3 {
    color: var(--qs-p);
    font-size: .9rem;
    margin: 0 0 .4rem;
    font-family: 'JetBrains Mono', monospace;
}
.qs-feat p { color: var(--qs-tm); font-size: .82rem; margin: 0; line-height: 1.5; }

/* Volatility table */
.qs-vol-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.qs-vol-table th {
    background: var(--qs-s2);
    color: var(--qs-p);
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: .5rem .8rem;
    text-align: left;
    border-bottom: 1px solid var(--qs-b);
}
.qs-vol-table td {
    padding: .55rem .8rem;
    border-bottom: 1px solid var(--qs-b);
    color: var(--qs-th);
}
.qs-vol-elevated { color: var(--qs-loss); font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.qs-vol-moderate { color: var(--qs-warn); font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.qs-vol-subdued  { color: var(--qs-profit); font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* Buttons */
.stButton > button {
    background: var(--qs-s2);
    border: 1px solid var(--qs-b);
    color: var(--qs-th);
    border-radius: 6px;
    transition: all .15s ease;
}
.stButton > button:hover {
    border-color: var(--qs-p);
    color: var(--qs-p);
    background: rgba(105,240,174,.08);
}
.stButton > button[kind="primary"] {
    background: var(--qs-p);
    color: #000;
    border: none;
    font-weight: 700;
}
.stButton > button[kind="primary"]:hover {
    background: var(--qs-pv);
    color: #000;
}

/* Inputs */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: var(--qs-s1) !important;
    border-color: var(--qs-b) !important;
    color: var(--qs-th) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: var(--qs-s1) !important;
    border: 1px solid var(--qs-b) !important;
    border-left: 3px solid var(--qs-sec) !important;
    border-radius: 8px !important;
}
[data-testid="stCaptionContainer"] { color: var(--qs-tm) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--qs-s1); border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: var(--qs-tm); }
.stTabs [aria-selected="true"] { color: var(--qs-p) !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state — pre-populate with first demo on cold start
# ---------------------------------------------------------------------------

_first_demo_key = list(DEMO_SCENARIOS.keys())[0]
_first_demo = DEMO_SCENARIOS[_first_demo_key]

if "transcript" not in st.session_state:
    st.session_state["transcript"] = _first_demo["transcript"]
if "news_raw" not in st.session_state:
    st.session_state["news_raw"] = "\n\n".join(_first_demo["news_items"])
if "context" not in st.session_state:
    st.session_state["context"] = _first_demo.get("context", "")
if "scenario_select" not in st.session_state:
    st.session_state["scenario_select"] = _first_demo_key

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="qs-brand"><h2>🛡️ RiskLabs</h2>'
        '<p>LLM Feature Extraction · Cao et al. 2024-2025</p></div>',
        unsafe_allow_html=True,
    )

    # Auth
    st.markdown('<div class="qs-sec">🔑 Authentication</div>', unsafe_allow_html=True)
    secret_key = load_api_key()
    if secret_key:
        api_key = secret_key
        st.success("Loaded from secrets / env", icon="🔒")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )

    # Model
    st.markdown('<div class="qs-sec">🤖 Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "LLM Model",
        options=["gpt-5.4-mini", "gpt-5.4-mini", "gpt-5.4-2026-03-05"],
        index=0,
        label_visibility="collapsed",
    )

    # Scenario selector
    st.markdown('<div class="qs-sec">📋 Demo Scenarios</div>', unsafe_allow_html=True)
    scenario_key = st.selectbox(
        "Scenario",
        options=list(DEMO_SCENARIOS.keys()),
        key="scenario_select",
        label_visibility="collapsed",
    )
    st.caption(DEMO_SCENARIOS[scenario_key]["description"])

    col_load, col_clear = st.columns(2)
    with col_load:
        if st.button("📥 Load", use_container_width=True, type="primary"):
            sc = DEMO_SCENARIOS[scenario_key]
            st.session_state["transcript"] = sc["transcript"]
            st.session_state["news_raw"] = "\n\n".join(sc["news_items"])
            st.session_state["context"] = sc.get("context", "")
            st.session_state.pop("result", None)
            st.rerun()
    with col_clear:
        if st.button("🧹 Clear", use_container_width=True):
            for k in ("transcript", "news_raw", "context", "result"):
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("📄 Cao et al. (2024-2025) · arXiv:2404.07452")
    st.caption("Reproduces LLM feature extraction (Figs. 2 & 3). Qualitative output only — NOT a numerical volatility predictor.")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="qs-hero">'
    "<h1>🛡️ RiskLabs Analyzer</h1>"
    '<div class="tag">LLM feature extraction · qualitative risk</div>'
    '<div class="ref">Cao et al. · arXiv:2404.07452 · 2024-2025</div>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API gate
# ---------------------------------------------------------------------------

if not api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to begin.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="qs-feat"><div class="icon">🗣️</div><h3>ECC ANALYZER</h3>'
            "<p>Hierarchical summarization of earnings call transcripts → structured risk flags, "
            "Question Bank answers, confidence signals (paper Fig. 2)</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="qs-feat"><div class="icon">📰</div><h3>NEWS ENRICHMENT</h3>'
            "<p>Four-step pipeline: sentiment → financial performance → regulatory → innovation "
            "for each news item in one batch LLM call (paper Fig. 3)</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="qs-feat"><div class="icon">🎯</div><h3>RISK SYNTHESIS</h3>'
            "<p>Combines ECC + news signals into risk score 0-100, risk level, "
            "4-horizon volatility outlook, and qualitative narrative</p></div>",
            unsafe_allow_html=True,
        )
    st.stop()

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------

st.markdown("#### Inputs")

col_l, col_r = st.columns([3, 2])

with col_l:
    transcript_val = st.text_area(
        "Earnings call transcript (paste)",
        key="transcript",
        height=280,
        placeholder="Paste the earnings call transcript here...",
    )

with col_r:
    news_raw_val = st.text_area(
        "News articles (one item per block, separate with blank lines)",
        key="news_raw",
        height=200,
        placeholder="Paste news snippets here, separated by blank lines...",
    )
    context_val = st.text_input(
        "Market context (optional — VIX level, macro environment, sector backdrop)",
        key="context",
        placeholder="e.g. VIX at 22, tech sector under pressure...",
    )

# Parse news
news_items = parse_news_text(st.session_state.get("news_raw", ""))

transcript_text = st.session_state.get("transcript", "").strip()

analyze_btn = st.button(
    "🔬 Analyze risk",
    type="primary",
    disabled=(not transcript_text),
    use_container_width=False,
)

if not transcript_text:
    st.caption("Paste a transcript above to enable analysis.")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if analyze_btn and transcript_text:
    llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    context_text = st.session_state.get("context", "")

    try:
        with st.status("Running RiskLabs pipeline...", expanded=True) as status:
            st.write("📝 Analyzing earnings call...")
            ecc_result = analyze_ecc(transcript_text, llm)
            st.write(
                f"   Found {len(ecc_result.risk_flags)} risk flags, "
                f"{len(ecc_result.confidence_signals)} confidence signals"
            )

            st.write("📰 Enriching news...")
            news_result = enrich_news(news_items, llm) if news_items else None
            n_news = len(news_result.signals) if news_result else 0
            st.write(f"   Processed {n_news} news items")

            st.write("🎯 Synthesizing risk assessment...")
            from risklabs_pipeline import NewsEnrichment
            risk_result = synthesize_risk(
                ecc_result,
                news_result if news_result else NewsEnrichment(signals=[]),
                context_text,
                llm,
            )
            st.write(
                f"   Risk score: {risk_result.risk_score}/100 — {risk_result.risk_level}"
            )

            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state["result"] = {
            "ecc": ecc_result,
            "news": news_result,
            "risk": risk_result,
        }
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result_bundle = st.session_state.get("result")

if result_bundle:
    risk: object = result_bundle["risk"]
    ecc: object = result_bundle["ecc"]
    news: object = result_bundle["news"]

    score = risk.risk_score
    level = risk.risk_level
    label_text, label_narrative = interpret_risk_level(score)

    # Score color
    if score <= 35:
        score_color = "var(--qs-profit)"
    elif score <= 65:
        score_color = "var(--qs-warn)"
    else:
        score_color = "var(--qs-loss)"

    # Confidence chip class
    conf_cls = {"low": "conf-low", "medium": "conf-med", "high": "conf-high"}.get(
        risk.confidence, "conf-med"
    )

    # Build volatility chips
    vol_chips_html = ""
    for horizon, outlook_text in risk.volatility_outlook.items():
        tone = outlook_text.split("—")[0].strip().lower() if "—" in outlook_text else outlook_text.split(" ")[0].lower()
        chip_cls = "elevated" if "elevated" in tone else ("moderate" if "moderate" in tone else "subdued")
        vol_chips_html += (
            f'<div class="qs-chip {chip_cls}">'
            f'{horizon.upper()} <strong>{tone}</strong>'
            f"</div>"
        )

    st.markdown(
        f'<div class="qs-risk-card">'
        f'<div style="color:var(--qs-tm);font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;font-family:\'JetBrains Mono\',monospace;">RISK SCORE</div>'
        f'<div class="qs-score" style="color:{score_color};">{score}</div>'
        f'<div class="qs-level" style="color:{score_color};">{level}</div>'
        f'<div class="qs-row">'
        f'{vol_chips_html}'
        f'<div class="qs-chip {conf_cls}">confidence <strong>{risk.confidence}</strong></div>'
        f"</div>"
        f'<div class="qs-narrative">{risk.narrative}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Tabs
    tab_ecc, tab_news, tab_synth = st.tabs(
        ["🗣️ ECC Analysis", "📰 News Signals", "🎯 Synthesis"]
    )

    # ------------------------------------------------------------------
    # ECC Tab
    # ------------------------------------------------------------------
    with tab_ecc:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Overview**")
            st.markdown(
                f'<div class="qs-card"><p>{ecc.overview_summary}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Financial Performance**")
            st.markdown(
                f'<div class="qs-card"><p>{ecc.performance_summary}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Key Topics**")
            st.markdown(
                f'<div class="qs-card"><p>{ecc.topic_summary}</p></div>',
                unsafe_allow_html=True,
            )

        with col_b:
            with st.expander("Question Bank — Cash Flow Drivers"):
                st.markdown(ecc.cash_flow_drivers)
            with st.expander("Question Bank — Rising Cost Impact"):
                st.markdown(ecc.rising_cost_impact)
            with st.expander("Question Bank — Dividend Outlook"):
                st.markdown(ecc.dividend_outlook)
            with st.expander("Innovation Analysis"):
                st.markdown(ecc.innovation_analysis)

        col_flags, col_sigs = st.columns(2)
        with col_flags:
            st.markdown("**Risk Flags**")
            if ecc.risk_flags:
                flags_html = "".join(
                    f'<li class="qs-risk-bullet">{f}</li>' for f in ecc.risk_flags
                )
                st.markdown(
                    f'<div class="qs-card"><ul style="margin:0;padding-left:1.2rem;">{flags_html}</ul></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="qs-card"><p style="color:var(--qs-profit)">No risk flags detected</p></div>',
                    unsafe_allow_html=True,
                )

        with col_sigs:
            st.markdown("**Confidence Signals**")
            if ecc.confidence_signals:
                sigs_html = "".join(
                    f'<li class="qs-ok-bullet">{s}</li>' for s in ecc.confidence_signals
                )
                st.markdown(
                    f'<div class="qs-card"><ul style="margin:0;padding-left:1.2rem;">{sigs_html}</ul></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="qs-card"><p style="color:var(--qs-tm)">No confidence signals detected</p></div>',
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------------
    # News Tab
    # ------------------------------------------------------------------
    with tab_news:
        if news and news.signals:
            # Emoji mappers
            _sent_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
            _fin_emoji = {"increasing": "↑", "decreasing": "↓", "stable": "→", "NaN": "—"}
            _reg_emoji = {
                "fraud": "🚨",
                "lawsuit": "⚖️",
                "antitrust": "🏛️",
                "quality_concerns": "⚠️",
                "none": "✅",
                "NaN": "—",
            }
            _inn_emoji = {
                "new_product": "🚀",
                "new_patent": "📜",
                "R&D": "🔬",
                "none": "—",
                "NaN": "—",
            }

            rows = []
            for sig in news.signals:
                rows.append(
                    {
                        "Headline": sig.headline[:80] + ("..." if len(sig.headline) > 80 else ""),
                        "Sentiment": f"{_sent_emoji.get(sig.sentiment, '')} {sig.sentiment}",
                        "Financial": f"{_fin_emoji.get(sig.financial_performance, '—')} {sig.financial_performance}",
                        "Regulatory": f"{_reg_emoji.get(sig.regulatory_issue, '—')} {sig.regulatory_issue}",
                        "Innovation": f"{_inn_emoji.get(sig.innovation_signal, '—')} {sig.innovation_signal}",
                        "Market Response": sig.market_response_prediction,
                    }
                )

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")
            with st.expander("Raw news items"):
                for i, item in enumerate(news_items):
                    st.markdown(f"**Item {i+1}**")
                    st.markdown(
                        f'<div class="qs-card"><p style="font-size:.82rem;">{item}</p></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No news items provided or analyzed.")

    # ------------------------------------------------------------------
    # Synthesis Tab
    # ------------------------------------------------------------------
    with tab_synth:
        col_pf, col_mf = st.columns(2)

        with col_pf:
            st.markdown("**Primary Risk Factors**")
            if risk.primary_risk_factors:
                items_html = "".join(
                    f'<li class="qs-risk-bullet" style="margin-bottom:.35rem;">{f}</li>'
                    for f in risk.primary_risk_factors
                )
                st.markdown(
                    f'<div class="qs-card"><ul style="margin:0;padding-left:1.2rem;">{items_html}</ul></div>',
                    unsafe_allow_html=True,
                )

        with col_mf:
            st.markdown("**Mitigating Factors**")
            if risk.mitigating_factors:
                items_html = "".join(
                    f'<li class="qs-ok-bullet" style="margin-bottom:.35rem;">{f}</li>'
                    for f in risk.mitigating_factors
                )
                st.markdown(
                    f'<div class="qs-card"><ul style="margin:0;padding-left:1.2rem;">{items_html}</ul></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="qs-card"><p style="color:var(--qs-tm)">No significant mitigating factors identified</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("**Volatility Outlook**")
        vol_rows = []
        _vol_labels = {"3d": "3-Day", "7d": "7-Day", "15d": "15-Day", "30d": "30-Day"}
        for horizon, outlook_text in risk.volatility_outlook.items():
            if "—" in outlook_text:
                parts = outlook_text.split("—", 1)
                tone = parts[0].strip().lower()
                reason = parts[1].strip() if len(parts) > 1 else ""
            else:
                tone = outlook_text.split(" ")[0].lower()
                reason = outlook_text
            vol_rows.append(
                {
                    "Horizon": _vol_labels.get(horizon, horizon),
                    "Outlook": tone.upper(),
                    "Reasoning": reason,
                }
            )

        if vol_rows:
            def color_outlook(val: str) -> str:
                v = val.lower()
                if "elevated" in v:
                    return "color: #FF5252; font-weight: 700"
                elif "moderate" in v:
                    return "color: #FFB74D; font-weight: 700"
                return "color: #00E676; font-weight: 700"

            vol_df = pd.DataFrame(vol_rows)
            st.dataframe(
                vol_df.style.map(color_outlook, subset=["Outlook"]),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Risk Narrative**")
        st.markdown(
            f'<div class="qs-card"><p>{risk.narrative}</p></div>',
            unsafe_allow_html=True,
        )

        # Interpretation footer
        st.markdown(
            f'<div class="qs-card" style="margin-top:1rem;">'
            f'<h4>Assessment Calibration</h4>'
            f'<p>{label_narrative}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
