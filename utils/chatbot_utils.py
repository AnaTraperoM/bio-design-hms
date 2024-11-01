import json
import streamlit as st

from langchain.schema import StrOutputParser

from utils.llm_utils import refresh_db
from utils.llm_utils import create_llm, get_prompt
from utils.google_sheet_utils import (
    get_case_sheet_as_dict, 
    get_observation_sheet_as_dict,
    cases_related_to_observations,
    observations_related_to_cases
)

def fetch_similar_data(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform similarity search using Pinecone
    updated_observations_db = refresh_db(namespace_to_refresh="observations")
    semantically_related_observations = updated_observations_db.similarity_search(prompt, k=3)

    cases_from_observations = cases_related_to_observations(semantically_related_observations)

    updated_cases_db = refresh_db(namespace_to_refresh="cases")
    semantically_related_cases = updated_cases_db.similarity_search(prompt, k=3)

    observations_from_cases = observations_related_to_cases(semantically_related_cases)

    return {"question": prompt, 
            "semantically_related_observations": semantically_related_observations,
            "cases_from_observations": cases_from_observations,
            "semantically_related_cases": semantically_related_cases,
            "observations_from_cases": observations_from_cases}

def fetch_real_time_gsheets_data(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    return {"question": prompt, 
            "semantically_related_observations": json.dumps(get_observation_sheet_as_dict()),
            "semantically_related_cases": json.dumps(get_case_sheet_as_dict()),
            "cases_from_observations": "None",
            "observations_from_cases": "None"
            }

def get_chat_response(user_input):

    llm = create_llm()
    observation_chat_chain = get_prompt() | llm | StrOutputParser()

    new_output = observation_chat_chain.invoke(fetch_real_time_gsheets_data(user_input),)
    
    return new_output



def update_session(output):
    # Update the conversation history
    # st.session_state.messages.append({"role": "assistant", "content": output})
    # st.write(st.session_state.messages)

    # Display the response
    with st.chat_message("assistant"):
        st.markdown(output)

    # Store chat in the current sheet
    st.session_state.chat_sheet.append_row([st.session_state.messages[-2].content, st.session_state.messages[-1].content])
