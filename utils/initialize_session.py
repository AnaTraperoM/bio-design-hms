import streamlit as st
from utils.google_sheet_utils import create_new_chat_sheet
from utils.chatbot_parameters import SYSTEM_PROMPT

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def initialize_investigator_session():
    # # Create a new sheet for the chat thread if not already created
    if "chat_sheet" not in st.session_state:
        st.session_state.chat_sheet = create_new_chat_sheet()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage(content=SYSTEM_PROMPT))

    # Display previous messages
    for message in st.session_state.messages:
        if not isinstance(message, SystemMessage):
            with st.chat_message(message.role):
                st.markdown(message.content)