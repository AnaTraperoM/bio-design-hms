import streamlit as st

from langchain.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 

from utils.login_utils import check_if_already_logged_in
from utils.page_formatting import add_investigator_formatting
from utils.initialize_session import initialize_investigator_session
from utils.chatbot_utils import update_session, get_chat_response

check_if_already_logged_in()
add_investigator_formatting()
initialize_investigator_session()

# Handle new input
if user_input := st.chat_input("What would you like to ask ?"):

    # show the user input
    with st.chat_message("user"):
        st.markdown(user_input)

    # create the chat messages for the llm
    if user_input is not None and user_input.strip()!="":
        # ai_response = st.write_stream(get_chat_response(user_input))
        ai_response = get_chat_response(user_input)

    # show the ai response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=ai_response))

    update_session(ai_response)

st.markdown("---")

# Spacer to push the button to the bottom
st.write(" " * 50)
