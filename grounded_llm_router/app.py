import io
import os
import sqlite3

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core import GroundedRouter, ingest, ingest_text, list_categories, DB_PATH, DOCS_DIR

load_dotenv()

st.set_page_config(page_title="NimbusStack Support Log", page_icon="—", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
.stApp { background-color: #F5F3ED; color: #1C1A16; }
section[data-testid="stSidebar"] { background-color: #EFEDE5; border-right: 1px solid #D8D4C8; }
section[data-testid="stSidebar"] * { font-family: 'Source Serif 4', Georgia, serif; }
.masthead { border-bottom: 2px solid #1C1A16; padding-bottom: 14px; margin-bottom: 6px; }
.masthead h1 { font-size: 26px; font-weight: 600; margin: 0; letter-spacing: 0.01em; }
.masthead .rule-note { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8577; margin-top: 4px; }
.lede { font-size: 16px; line-height: 1.6; max-width: 640px; color: #3A362E; margin: 20px 0 28px 0; }
.stTextInput input { background-color: transparent; border: none; border-bottom: 1px solid #1C1A16; border-radius: 0; color: #1C1A16; font-family: 'Source Serif 4', Georgia, serif; font-size: 17px; padding: 6px 2px; }
.stTextInput input:focus { box-shadow: none; border-bottom: 2px solid #1C1A16; }
.stButton button { background-color: transparent; color: #1C1A16; border: 1px solid #1C1A16; border-radius: 0; font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.04em; padding: 0.4rem 1.1rem; box-shadow: none; }
.stButton button:hover { background-color: #1C1A16; color: #F5F3ED; border-color: #1C1A16; }
.entry { border-top: 1px solid #D8D4C8; padding: 20px 0; }
.entry-no { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8577; }
.entry-q { font-style: italic; color: #57523F; margin: 4px 0 10px 0; font-size: 15px; }
.entry-a { font-size: 17px; line-height: 1.6; margin-bottom: 10px; }
.entry-a.refused { color: #7A2E1F; }
.entry-meta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8577; }
.entry-meta .status.grounded { color: #1C1A16; font-weight: 500; }
.entry-meta .status.refused { color: #7A2E1F; font-weight: 500; }
.ledger-row { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 12px; padding: 7px 0; border-bottom: 1px solid #E2DFD4; color: #57523F; }
.ledger-row .status-mark.refused { color: #7A2E1F; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8A8577; letter-spacing: 0.05em; margin: 20px 0 8px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_router(api_key: str):
    router = GroundedRouter(api_key=api_key or None)
    if not router.collection.get()["ids"]:
        ingest()
    return router


if "history" not in st.session_state:
    st.session_state.history = []
if "entry_count" not in st.session_state:
    st.session_state.entry_count = 0

with st.sidebar:
    st.markdown("<div class='section-label'>configuration</div>", unsafe_allow_html=True)
    api_key_input = st.text_input("Gemini API key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    router = get_router(api_key_input)

    if st.button("Rebuild built-in index"):
        with st.spinner("Re-reading docs/*.md"):
            n = ingest()
        st.success(f"{n} passages indexed")

    st.markdown("<div class='section-label'>search scope</div>", unsafe_allow_html=True)
    all_categories = ["All"] + list_categories()
    scope = st.selectbox("Category", all_categories, label_visibility="collapsed")

    with st.expander("Advanced retrieval settings"):
        top_k = st.slider("Passages retrieved per query", 1, 8, 3)
        threshold_override = st.slider(
            "Refusal distance threshold", 0.5, 1.5, 1.05, 0.05,
            help="Lower = stricter (refuses more often). Higher = more lenient.",
        )

    st.markdown("<div class='section-label'>add documents</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload .md or .txt files", type=["md", "txt"], accept_multiple_files=True)
    if uploaded and st.button("Add to knowledge base"):
        total_chunks = 0
        for f in uploaded:
            text = f.read().decode("utf-8", errors="ignore")
            category = f"uploaded: {f.name}"
            total_chunks += ingest_text(f.name, text, category)
        st.success(f"Added {len(uploaded)} file(s), {total_chunks} chunks. New categories are now searchable.")
        st.rerun()

    st.markdown("<div class='section-label'>ledger — last 10</div>", unsafe_allow_html=True)
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        log_df = pd.read_sql_query(
            "SELECT timestamp, query, grounded, category FROM query_log ORDER BY id DESC LIMIT 10", conn
        )
        conn.close()
        if len(log_df):
            for _, r in log_df.iterrows():
                mark = "grounded" if r["grounded"] else "refused"
                mark_class = "" if r["grounded"] else "refused"
                short_q = (r["query"][:26] + "…") if len(r["query"]) > 26 else r["query"]
                time_short = r["timestamp"][11:16] if isinstance(r["timestamp"], str) else ""
                st.markdown(
                    f"<div class='ledger-row'><span>{time_short} — {short_q}</span>"
                    f"<span class='status-mark {mark_class}'>{mark}</span></div>",
                    unsafe_allow_html=True,
                )

            grounded_rate = log_df["grounded"].mean() * 100
            c1, c2 = st.columns(2)
            c1.metric("Grounded", f"{grounded_rate:.0f}%")
            c2.metric("Entries", len(log_df))

            full_conn = sqlite3.connect(DB_PATH)
            full_df = pd.read_sql_query("SELECT * FROM query_log ORDER BY id DESC", full_conn)
            full_conn.close()
            csv_buffer = io.StringIO()
            full_df.to_csv(csv_buffer, index=False)
            st.download_button("Export full log (CSV)", csv_buffer.getvalue(), "query_log.csv", "text/csv")
        else:
            st.markdown("<div style='color:#8A8577; font-size:12px;'>Nothing recorded yet.</div>", unsafe_allow_html=True)


st.markdown("""
<div class="masthead">
    <h1>NimbusStack Support Log</h1>
    <div class="rule-note">pricing · security · sla · features · integrations · admin · onboarding · roadmap</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="lede">
Ask anything about NimbusStack. Narrow the search scope in the sidebar to a
specific topic, or leave it on "All" to search everything, including any
documents you've uploaded. Every answer is checked against the source docs
first -- if nothing backs it up closely enough, it's logged as a refusal
rather than a guess.
</div>
""", unsafe_allow_html=True)

if not list(DOCS_DIR.glob("*.md")):
    st.warning("No built-in source documents found in docs/.")

query = st.text_input("query", placeholder="what's the uptime SLA for the growth tier", label_visibility="collapsed")
ask = st.button("Ask")

if ask and query:
    with st.spinner("Checking the record…"):
        result = router.answer(query, k=top_k, category=scope, distance_threshold=threshold_override)
    st.session_state.entry_count += 1
    st.session_state.history.insert(0, {
        "n": st.session_state.entry_count, "query": query, "result": result,
    })

for item in st.session_state.history:
    result = item["result"]
    css_class = "" if result["grounded"] else "refused"
    status_word = "grounded" if result["grounded"] else "refused"
    model_str = result["model_used"] or "—"
    sources_str = ", ".join(result["sources"]) if result["sources"] else "none"

    st.markdown(f"""
    <div class="entry">
        <div class="entry-no">entry {item['n']:03d} · scope: {result['category']}</div>
        <div class="entry-q">{item['query']}</div>
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
