import streamlit as st
from utils.google_sheet_utils import create_new_chat_sheet
from utils.chatbot_parameters import SYSTEM_PROMPT

from utils.google_sheet_utils import sync_with_pinecone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def initialize_investigator_session():
    # # Create a new sheet for the chat thread if not already created
    if "chat_sheet" not in st.session_state:
        st.session_state.chat_sheet = create_new_chat_sheet()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "observation_sheet_name" not in st.session_state:
        st.session_state.observation_sheet_name = "Copy of 2024 Healthtech Identify Log"
       
    if "observation_namespace" not in st.session_state:
        st.session_state.observation_namespace = "observations_temp_v2"
        sync_with_pinecone()

    # Display previous messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)