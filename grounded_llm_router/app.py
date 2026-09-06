import os
import sqlite3

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core import GroundedRouter, ingest, DB_PATH, DOCS_DIR

load_dotenv()

st.set_page_config(page_title="Grounded LLM Router", page_icon="\U0001F9ED", layout="wide")

st.title("Grounded LLM Router")
st.caption(
    "A cost-aware support chatbot for a fictional SaaS (NimbusStack) that "
    "refuses to guess when it isn't grounded in the docs, and routes queries "
    "to a cheap or strong model based on complexity and retrieval confidence."
)

with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input(
        "Gemini API key (optional — get one free at aistudio.google.com)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
    )
    if st.button("Re-ingest documents"):
        with st.spinner("Chunking and embedding docs/*.md ..."):
            n = ingest()
        st.success(f"Ingested {n} chunks.")

    st.divider()
    st.header("Query log")
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT timestamp, query, grounded, model_used, "
            "ROUND(est_cost_usd, 6) as est_cost_usd, "
            "ROUND(latency_ms, 0) as latency_ms "
            "FROM query_log ORDER BY id DESC LIMIT 20",
            conn,
        )
        conn.close()
        st.dataframe(df, use_container_width=True, hide_index=True)
        if len(df):
            grounded_rate = df["grounded"].mean() * 100
            st.metric("Grounded-answer rate (last 20)", f"{grounded_rate:.0f}%")
    else:
        st.info("No queries logged yet.")

if "router" not in st.session_state or st.session_state.get("_key") != api_key_input:
    st.session_state.router = GroundedRouter(api_key=api_key_input or None)
    st.session_state._key = api_key_input

if not list(DOCS_DIR.glob("*.md")):
    st.warning("No docs found in docs/. Add some .md files and click 'Re-ingest documents'.")

query = st.text_input(
    "Ask a question about NimbusStack (pricing, security, SLA, or features):",
    placeholder="e.g. What's the uptime SLA for the Growth tier?",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask", type="primary")

if ask and query:
    with st.spinner("Retrieving context and generating answer..."):
        result = st.session_state.router.answer(query)

    if result["grounded"]:
        st.success(result["answer"])
        st.caption(
            f"Model: `{result['model_used']}` \u00b7 "
            f"Est. cost: ${result['est_cost_usd']:.6f} \u00b7 "
            f"Latency: {result['latency_ms']:.0f} ms \u00b7 "
            f"Retrieval distance: {result['best_distance']:.3f}"
        )
        with st.expander("Sources"):
            for s in result["sources"]:
                st.write(f"- {s}")
    else:
        st.error(result["answer"])
        st.caption(f"Best retrieval distance: {result['best_distance']:.3f}")

st.divider()
st.caption(
    "Try an out-of-scope question too, e.g. 'What's the weather today?' or "
    "'Does NimbusStack support quantum computing workflows?' — it should "
    "refuse instead of making something up."
)
