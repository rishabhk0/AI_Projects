import os
import sqlite3

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core import GroundedRouter, ingest, DB_PATH, DOCS_DIR

load_dotenv()

st.set_page_config(page_title="NimbusStack Support Log", page_icon="—", layout="wide")

# ---------------------------------------------------------------------------
# Style — ledger / paper aesthetic. Off-white page, black serif ink, hairline
# rules, no rounded corners, no shadows. Red-brown ink is reserved entirely
# for refusals, the way a correction mark stands out on a page of otherwise
# ordinary handwriting.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
}

.stApp {
    background-color: #F5F3ED;
    color: #1C1A16;
}

section[data-testid="stSidebar"] {
    background-color: #EFEDE5;
    border-right: 1px solid #D8D4C8;
}
section[data-testid="stSidebar"] * {
    font-family: 'Source Serif 4', Georgia, serif;
}

.masthead {
    border-bottom: 2px solid #1C1A16;
    padding-bottom: 14px;
    margin-bottom: 6px;
}
.masthead h1 {
    font-size: 26px;
    font-weight: 600;
    font-style: normal;
    margin: 0;
    letter-spacing: 0.01em;
}
.masthead .rule-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8A8577;
    margin-top: 4px;
}

.lede {
    font-size: 16px;
    line-height: 1.6;
    max-width: 620px;
    color: #3A362E;
    margin: 20px 0 28px 0;
}

.stTextInput input {
    background-color: transparent;
    border: none;
    border-bottom: 1px solid #1C1A16;
    border-radius: 0;
    color: #1C1A16;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 17px;
    padding: 6px 2px;
}
.stTextInput input:focus {
    box-shadow: none;
    border-bottom: 2px solid #1C1A16;
}

.stButton button {
    background-color: transparent;
    color: #1C1A16;
    border: 1px solid #1C1A16;
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.04em;
    padding: 0.4rem 1.1rem;
    box-shadow: none;
}
.stButton button:hover {
    background-color: #1C1A16;
    color: #F5F3ED;
    border-color: #1C1A16;
}

.entry {
    border-top: 1px solid #D8D4C8;
    padding: 20px 0;
}
.entry-no {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8A8577;
}
.entry-q {
    font-style: italic;
    color: #57523F;
    margin: 4px 0 10px 0;
    font-size: 15px;
}
.entry-a {
    font-size: 17px;
    line-height: 1.6;
    margin-bottom: 10px;
}
.entry-a.refused {
    color: #7A2E1F;
}
.entry-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8A8577;
}
.entry-meta .status.grounded { color: #1C1A16; font-weight: 500; }
.entry-meta .status.refused { color: #7A2E1F; font-weight: 500; }

.ledger-row {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 7px 0;
    border-bottom: 1px solid #E2DFD4;
    color: #57523F;
}
.ledger-row .status-mark.refused { color: #7A2E1F; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_router(api_key: str):
    router = GroundedRouter(api_key=api_key or None)
    if not router.collection.get()["ids"]:
        ingest()
    return router


if "entry_count" not in st.session_state:
    st.session_state.entry_count = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None
    st.session_state.last_query = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family: IBM Plex Mono, monospace; font-size: 11px; color: #8A8577; margin-bottom: 8px;'>configuration</div>", unsafe_allow_html=True)
    api_key_input = st.text_input("Gemini API key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

    if st.button("Rebuild index"):
        with st.spinner("Re-reading docs/*.md"):
            n = ingest()
        st.success(f"{n} passages indexed")

    st.markdown("<hr style='border-color: #D8D4C8; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: IBM Plex Mono, monospace; font-size: 11px; color: #8A8577; margin-bottom: 8px;'>ledger — last 10</div>", unsafe_allow_html=True)

    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        log_df = pd.read_sql_query(
            "SELECT timestamp, query, grounded FROM query_log ORDER BY id DESC LIMIT 10", conn
        )
        conn.close()
        if len(log_df):
            for _, r in log_df.iterrows():
                mark = "grounded" if r["grounded"] else "refused"
                mark_class = "" if r["grounded"] else "refused"
                short_q = (r["query"][:30] + "…") if len(r["query"]) > 30 else r["query"]
                time_short = r["timestamp"][11:16] if isinstance(r["timestamp"], str) else ""
                st.markdown(
                    f"<div class='ledger-row'><span>{time_short} — {short_q}</span>"
                    f"<span class='status-mark {mark_class}'>{mark}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div style='color:#8A8577; font-size:12px;'>Nothing recorded yet.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("""
<div class="masthead">
    <h1>NimbusStack Support Log</h1>
    <div class="rule-note">pricing · security · sla · features</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="lede">
Ask anything about pricing, security posture, SLA terms, or product features.
Every answer is checked against the source documents first — if nothing
backs it up closely enough, it goes in the log as a refusal rather than
a guess.
</div>
""", unsafe_allow_html=True)

if not list(DOCS_DIR.glob("*.md")):
    st.warning("No source documents found. Add .md files to docs/ and rebuild the index.")

query = st.text_input("query", placeholder="what's the uptime SLA for the growth tier", label_visibility="collapsed")
ask = st.button("Ask")

if ask and query:
    router = get_router(api_key_input)
    with st.spinner("Checking the record…"):
        result = router.answer(query)
    st.session_state.entry_count += 1
    st.session_state.last_result = result
    st.session_state.last_query = query

if st.session_state.last_result:
    result = st.session_state.last_result
    css_class = "" if result["grounded"] else "refused"
    status_word = "grounded" if result["grounded"] else "refused"
    model_str = result["model_used"] or "—"
    sources_str = ", ".join(result["sources"]) if result["sources"] else "none"

    st.markdown(f"""
    <div class="entry">
        <div class="entry-no">entry {st.session_state.entry_count:03d}</div>
        <div class="entry-q">{st.session_state.last_query}</div>
        <div class="entry-a {css_class}">{result['answer']}</div>
        <div class="entry-meta">
            <span class="status {css_class}">{status_word}</span>
            &nbsp;·&nbsp; model {model_str}
            &nbsp;·&nbsp; distance {result['best_distance']:.3f}
            &nbsp;·&nbsp; cost ${result['est_cost_usd']:.6f}
            &nbsp;·&nbsp; {result['latency_ms']:.0f}ms
            &nbsp;·&nbsp; sources: {sources_str}
        </div>
    </div>
    """, unsafe_allow_html=True)
