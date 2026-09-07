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

.stApp { background-color: #1B1917; color: #E8E4DA; }

section[data-testid="stSidebar"] { background-color: #221F1B; border-right: 1px solid #3A362C; }
section[data-testid="stSidebar"] * { font-family: 'Source Serif 4', Georgia, serif; color: #E8E4DA; }

.masthead { border-bottom: 2px solid #E8E4DA; padding-bottom: 14px; margin-bottom: 6px; }
.masthead h1 { font-size: 26px; font-weight: 600; margin: 0; letter-spacing: 0.01em; color: #E8E4DA; }
.masthead .rule-note { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #9A9484; margin-top: 4px; }

.lede { font-size: 16px; line-height: 1.6; max-width: 640px; color: #C9C3B4; margin: 20px 0 28px 0; }

.stTextInput input {
    background-color: transparent;
    border: none;
    border-bottom: 1px solid #E8E4DA;
    border-radius: 0;
    color: #E8E4DA;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 17px;
    padding: 6px 2px;
}
.stTextInput input:focus { box-shadow: none; border-bottom: 2px solid #E8E4DA; }
.stTextInput input::placeholder { color: #6B6656; }

.stButton button {
    background-color: transparent;
    color: #E8E4DA;
    border: 1px solid #E8E4DA;
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.04em;
    padding: 0.4rem 1.1rem;
    box-shadow: none;
}
.stButton button:hover { background-color: #E8E4DA; color: #1B1917; border-color: #E8E4DA; }

.entry { border-top: 1px solid #3A362C; padding: 20px 0; }
.entry-no { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #9A9484; }
.entry-q { font-style: italic; color: #B8B2A0; margin: 4px 0 10px 0; font-size: 15px; }
.entry-a { font-size: 17px; line-height: 1.6; margin-bottom: 10px; color: #E8E4DA; }
.entry-a.refused { color: #E0855F; }
.entry-meta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #9A9484; }
.entry-meta .status.grounded { color: #E8E4DA; font-weight: 500; }
.entry-meta .status.refused { color: #E0855F; font-weight: 500; }

.
