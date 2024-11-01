import streamlit as st

from langchain.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 

from utils.login_utils import check_if_already_logged_in
from utils.google_sheet_utils import create_new_chat_sheet, get_case_descriptions_from_case_ids
from utils.page_formatting import add_investigator_formatting
from utils.initialize_session import initialize_investigator_session
from utils.chatbot_utils import fetch_similar_data, update_session, fetch_real_time_gsheets_data, get_chat_response
from utils.chatbot_parameters import SYSTEM_PROMPT

check_if_already_logged_in()
add_investigator_formatting()
initialize_investigator_session()

# llm = create_llm()
# observation_chat_chain = get_prompt() | llm | StrOutputParser()

# st.session.state.messages = []
# st.session_state.messages.append({"role": "system", "content": SYSTEM_PROMPT})

# Handle new input
if user_input := st.chat_input("What would you like to ask me?"):

    # create the chat messages for the llm
    if user_input is not None and user_input.strip()!="":
        ai_response = get_chat_response(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=ai_response))


    # # with get_openai_callback() as cb:
    # new_output = observation_chat_chain.invoke(fetch_real_time_gsheets_data(prompt),)
    # # new_output = observation_chat_chain.invoke(fetch_similar_data(prompt),)

    update_session(new_output)

st.markdown("---")

# Spacer to push the button to the bottom
st.write(" " * 50)
