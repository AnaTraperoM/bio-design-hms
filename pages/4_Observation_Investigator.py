import streamlit as st

from langchain.schema import StrOutputParser
from langchain.callbacks import get_openai_callback

from utils.login_utils import check_if_already_logged_in
from utils.google_sheet_utils import create_new_chat_sheet, get_case_descriptions_from_case_ids
from utils.llm_utils import create_llm, get_prompt
from utils.page_formatting import add_investigator_formatting
from utils.initialize_session import initialize_investigator_session
from utils.chatbot_utils import fetch_similar_data, update_session, fetch_real_time_gsheets_data

check_if_already_logged_in()
add_investigator_formatting()
initialize_investigator_session()

llm = create_llm()
observation_chat_chain = get_prompt() | llm | StrOutputParser()

# Handle new input
if prompt := st.chat_input("What would you like to ask?"):

    with get_openai_callback() as cb:
        new_output = observation_chat_chain.invoke(fetch_similar_data(prompt),)

    update_session(new_output)

st.markdown("---")

# Spacer to push the button to the bottom
st.write(" " * 50)
