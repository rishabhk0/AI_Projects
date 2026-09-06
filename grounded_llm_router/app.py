import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core import GroundedRouter, ingest, DB_PATH, DOCS_DIR

load_dotenv()

st.set_page_config(page_title="NimbusStack Query Console", page_icon="◈", layout="wide")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background-color: #14171C;
    color: #E8E6E1;
}

section[data-testid="stSidebar"] {
    background-color: #1A1E26;
    border-right: 1px solid #2A2F3A;
}

.console-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 4px;
}
.console-header h1 {
    font-size: 28px;
    font-weight: 700;
    margin: 0;
    color: #E8E6E1;
}
.console-header .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #6E7686;
    letter-spacing: 0.02em;
}
.console-sub {
    color: #8B92A0;
    font-size: 15px;
    max-width: 640px;
    line-height: 1.5;
    margin-bottom: 28px;
}

.stTextInput input {
    background-color: #1C2029;
    border: 1px solid #2A2F3A;
    color: #E8E6E1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    border-radius: 3px;
}
.stTextInput input:focus {
    border-color: #4FD1C5;
    box-shadow: none;
}

.stButton button {
    background-color: #4FD1C5;
    color: #0F1115;
    border: none;
    border-radius: 3px;
    font-weight: 500;
    padding: 0.5rem 1.4rem;
}
.stButton button:hover {
    background-color: #6EE0D5;
    color: #0F1115;
}

.readout {
    border: 1px solid #2A2F3A;
    border-radius: 4px;
    padding: 18px 20px;
    margin-top: 18px;
    background-color: #1C2029;
}
.readout.grounded { border-left: 3px solid #4FD1C5; }
.readout.refused { border-left: 3px solid #E8965A; }
.readout-answer {
    font-size: 16px;
    line-height: 1.55;
    margin-bottom: 14px;
}
.readout-meta {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8B92A0;
    border-top: 1px solid #2A2F3A;
    padding-top: 10px;
}
.readout-meta span b {
    color: #C7CCD6;
    font-weight: 500;
}

.log-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.log-table th {
    text-align: left;
    color: #6E7686;
    font-weight: 500;
    padding: 6px 8px;
    border-bottom: 1px solid #2A2F3A;
}
.log-table td {
    padding: 6px 8px;
    border-bottom: 1px solid #22262F;
    color: #C7CCD6;
}
.log-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached, process-level resource — loads the embedding model and Chroma
# collection once per server process, not once per visitor session. This is
# the fix for slow responses: without it, every new session reloaded the
# sentence-transformer model from scratch on a shared, CPU-only instance.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_router(api_key: str):
    router = GroundedRouter(api_key=api_key or None)
    if not router.collection.get()["ids"]:
        ingest()
    return router


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family: JetBrains Mono, monospace; font-size: 12px; color: #6E7686; letter-spacing: 0.05em; margin-bottom: 10px;'>CONFIG</div>", unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Free at aistudio.google.com",
    )

    if st.button("Rebuild index"):
        with st.spinner("Re-embedding docs/*.md"):
            n = ingest()
        st.success(f"{n} chunks indexed")

    st.markdown("<hr style='border-color: #2A2F3A; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: JetBrains Mono, monospace; font-size: 12px; color: #6E7686; letter-spacing: 0.05em; margin-bottom: 10px;'>RECENT QUERIES</div>", unsafe_allow_html=True)

    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        log_df = pd.read_sql_query(
            "SELECT timestamp, query, grounded, model_used, "
            "ROUND(est_cost_usd, 6) as cost, ROUND(latency_ms, 0) as latency_ms "
            "FROM query_log ORDER BY id DESC LIMIT 12",
            conn,
        )
        conn.close()

        if len(log_df):
            rows_html = ""
            for _, r in log_df.iterrows():
                dot_color = "#4FD1C5" if r["grounded"] else "#E8965A"
                short_q = (r["query"][:38] + "…") if len(r["query"]) > 38 else r["query"]
                time_short = r["timestamp"][11:19] if isinstance(r["timestamp"], str) else ""
                rows_html += f"""<tr>
                    <td><span class="log-dot" style="background-color:{dot_color}"></span>{time_short}</td>
                    <td>{short_q}</td>
                </tr>"""
            st.markdown(f"""
                <table class="log-table">
                    <tr><th>time</th><th>query</th></tr>
                    {rows_html}
                </table>
            """, unsafe_allow_html=True)

            grounded_rate = log_df["grounded"].mean() * 100
            st.markdown(f"<div style='margin-top:14px; font-family: JetBrains Mono, monospace; font-size: 12px; color: #6E7686;'>grounded rate, last {len(log_df)}: <b style=\"color:#C7CCD6\">{grounded_rate:.0f}%</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#6E7686; font-size:13px;'>No queries yet.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#6E7686; font-size:13px;'>No queries yet.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("""
<div class="console-header">
    <h1>NimbusStack Query Console</h1>
    <span class="tag">v1 · pricing · security · sla · features</span>
</div>
<div class="console-sub">
    Answers are checked against the product docs before anything is shown.
    If nothing in the docs backs up an answer closely enough, you'll get a
    refusal instead of a guess — try an unrelated question to see it happen.
</div>
""", unsafe_allow_html=True)

if not list(DOCS_DIR.glob("*.md")):
    st.warning("No docs found in docs/. Add .md files and rebuild the index from the sidebar.")

query = st.text_input(
    "query",
    placeholder="e.g. what's the uptime SLA for the growth tier",
    label_visibility="collapsed",
)
ask = st.button("Run query")

if ask and query:
    router = get_router(api_key_input)
    with st.spinner("Checking docs and querying model…"):
        result = router.answer(query)

    css_class = "grounded" if result["grounded"] else "refused"
    status_label = "GROUNDED" if result["grounded"] else "REFUSED"
    status_color = "#4FD1C5" if result["grounded"] else "#E8965A"

    sources_str = ", ".join(result["sources"]) if result["sources"] else "none"
    model_str = result["model_used"] or "n/a"

    st.markdown(f"""
    <div class="readout {css_class}">
        <div class="readout-answer">{result['answer']}</div>
        <div class="readout-meta">
            <span style="color:{status_color}; font-weight:500;">{status_label}</span>
            <span>model: <b>{model_str}</b></span>
            <span>distance: <b>{result['best_distance']:.3f}</b></span>
            <span>cost: <b>${result['est_cost_usd']:.6f}</b></span>
            <span>latency: <b>{result['latency_ms']:.0f}ms</b></span>
            <span>sources: <b>{sources_str}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
