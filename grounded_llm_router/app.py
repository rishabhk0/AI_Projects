import os

import streamlit as st
from dotenv import load_dotenv

from core import (
    GroundedRouter, ingest, ingest_text, list_categories,
    create_conversation, list_conversations, get_messages, add_message,
    export_all_messages_csv, init_db, DOCS_DIR,
)

load_dotenv()

# Must run before anything touches the database -- list_conversations() is
# called in the sidebar before GroundedRouter() is ever constructed, so
# relying on GroundedRouter.__init__() to create the tables was too late.
init_db()

st.set_page_config(page_title="NimbusStack", page_icon="—", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
.stApp { background-color: #1B1917; color: #E8E4DA; }

section[data-testid="stSidebar"] { background-color: #221F1B; border-right: 1px solid #3A362C; }
section[data-testid="stSidebar"] * { font-family: 'Source Serif 4', Georgia, serif; color: #E8E4DA; }

.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #9A9484; letter-spacing: 0.05em; margin: 20px 0 8px 0; }

.stButton button {
    background-color: transparent;
    color: #E8E4DA;
    border: 1px solid #3A362C;
    border-radius: 6px;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
    text-align: left;
    padding: 0.45rem 0.8rem;
    box-shadow: none;
    width: 100%;
}
.stButton button:hover { background-color: #2A261F; border-color: #E8E4DA; }

.new-chat-btn button {
    border: 1px solid #E8E4DA !important;
    font-weight: 500;
}

.conv-active button {
    background-color: #2A261F !important;
    border-color: #E8E4DA !important;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding-top: 18vh;
    text-align: center;
    color: #9A9484;
}
.empty-state h2 { color: #E8E4DA; font-weight: 600; font-size: 24px; margin-bottom: 8px; }
.empty-state p { max-width: 480px; font-size: 15px; line-height: 1.5; }

[data-testid="stChatMessage"] { background-color: transparent; }
.stChatInput textarea { font-family: 'Source Serif 4', Georgia, serif !important; }

.msg-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #9A9484;
    margin-top: 6px;
}
.msg-meta.refused { color: #E0855F; }

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #221F1B; border-color: #3A362C; color: #E8E4DA;
}
.stSlider label, .stSelectbox label { color: #E8E4DA !important; }

[data-testid="stIconMaterial"],
span[class*="material-symbols"],
span[class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_router():
    router = GroundedRouter(api_key=os.environ.get("GEMINI_API_KEY"))
    if not router.collection.get()["ids"]:
        ingest()
    return router


if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None  # None = pending new chat


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-size:18px; font-weight:600; margin-bottom:16px;'>NimbusStack</div>", unsafe_allow_html=True)

    st.markdown("<div class='new-chat-btn'>", unsafe_allow_html=True)
    if st.button("+ New chat", key="new_chat_btn"):
        st.session_state.current_conversation_id = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>chats</div>", unsafe_allow_html=True)
    conversations = list_conversations()
    if not conversations:
        st.markdown("<div style='color:#6B6656; font-size:13px;'>No chats yet.</div>", unsafe_allow_html=True)
    for conv in conversations:
        is_active = conv["id"] == st.session_state.current_conversation_id
        wrapper_class = "conv-active" if is_active else ""
        st.markdown(f"<div class='{wrapper_class}'>", unsafe_allow_html=True)
        if st.button(conv["title"] or "Untitled", key=f"conv_{conv['id']}"):
            st.session_state.current_conversation_id = conv["id"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    router = get_router()

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

    st.markdown("<div class='section-label'>export</div>", unsafe_allow_html=True)
    csv_data = export_all_messages_csv()
    st.download_button("Export all chats (CSV)", csv_data, "nimbusstack_chats.csv", "text/csv")


# ---------------------------------------------------------------------------
# Main -- chat area
# ---------------------------------------------------------------------------
if not list(DOCS_DIR.glob("*.md")):
    st.warning("No built-in source documents found in docs/.")

current_id = st.session_state.current_conversation_id
messages = get_messages(current_id) if current_id else []

if not messages:
    st.markdown("""
    <div class="empty-state">
        <h2>NimbusStack Support</h2>
        <p>Ask anything about pricing, security, SLA, features, integrations, admin,
        onboarding, or the roadmap. Attach a document with the + button to search
        it too. Answers are checked against the source docs first -- if nothing
        backs one up closely enough, you'll get an honest "I don't know" instead
        of a guess.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in messages:
        role = msg["role"]
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(msg["content"])
            if role == "assistant" and msg.get("model_used") is not None:
                refused_class = "" if msg["grounded"] else "refused"
                status_word = "grounded" if msg["grounded"] else "refused"
                st.markdown(
                    f"<div class='msg-meta {refused_class}'>{status_word} · "
                    f"model {msg['model_used']} · distance {msg['best_distance']:.3f} · "
                    f"cost ${msg['est_cost_usd']:.6f} · {msg['latency_ms']:.0f}ms</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Input -- native chat_input with built-in + attach button
# ---------------------------------------------------------------------------
prompt = st.chat_input(
    "Message NimbusStack…",
    accept_file="multiple",
    file_type=["md", "txt"],
)

if prompt:
    user_text = prompt.text.strip() if prompt.text else ""
    attached_files = prompt.files if prompt.files else []

    # Lazily create the conversation on first message, same as ChatGPT --
    # "New chat" doesn't appear in the sidebar list until something is sent.
    if current_id is None:
        title_seed = user_text or (attached_files[0].name if attached_files else "New chat")
        current_id = create_conversation(title_seed)
        st.session_state.current_conversation_id = current_id

    if attached_files:
        total_chunks = 0
        for f in attached_files:
            text = f.read().decode("utf-8", errors="ignore")
            category = f"uploaded: {f.name}"
            total_chunks += ingest_text(f.name, text, category)
        file_names = ", ".join(f.name for f in attached_files)
        add_message(current_id, "user", user_text or f"[Attached: {file_names}]")
        add_message(
            current_id, "assistant",
            f"Added {len(attached_files)} file(s) ({file_names}) to the knowledge base, "
            f"{total_chunks} chunks indexed and searchable now.",
        )

    if user_text:
        if not attached_files:
            add_message(current_id, "user", user_text)
        result = router.answer(user_text, k=top_k, category=scope, distance_threshold=threshold_override)
        add_message(current_id, "assistant", result["answer"], meta=result)

    st.rerun()