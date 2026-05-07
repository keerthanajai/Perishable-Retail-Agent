import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from agent.agent import DemandAgent

st.set_page_config(
    page_title="Demand Intelligence Agent",
    page_icon="🛒",
    layout="centered"
)

# --- Styling ---
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stChatMessage { border-radius: 12px; padding: 8px; }
        .sidebar-info { background-color: #e8f4f8; padding: 12px;
                        border-radius: 8px; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🛒 Demand Intelligence Agent")
st.caption("AI-powered demand forecasting for perishable retail — powered by XGBoost + Claude")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("📊 About This Agent")
    st.markdown("""
    <div class="sidebar-info">
    This agent uses a trained <b>XGBoost model</b> on 31M+ retail records
    to forecast demand and explain predictions using <b>SHAP</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔧 Try These Queries")
    example_queries = [
        "Should I stock up on bread at store 1, item 103665?",
        "What's the forecast for store 3, item 103665?",
        "Why is demand low for store 1, item 103665?",
        "What would happen if item 103665 at store 1 goes on promotion?",
        "Give me a full analysis for store 5, item 103665",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True):
            st.session_state.pending_query = q

    st.markdown("---")
    st.subheader("ℹ️ Data Info")
    st.markdown("""
    - **Stores:** 1–54
    - **Records:** 31.7M+
    - **Features:** 54 engineered
    - **Model:** XGBoost + SHAP
    - **Dataset:** Favorita Grocery (Ecuador)
    """)

    st.markdown("---")
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent.reset()
        st.rerun()

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = DemandAgent()

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# --- Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# --- Handle Sidebar Button Clicks ---
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing demand data..."):
            response = st.session_state.agent.chat(query)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# --- Chat Input ---
if prompt := st.chat_input("Ask about demand, forecasts, or inventory decisions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing demand data..."):
            response = st.session_state.agent.chat(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})