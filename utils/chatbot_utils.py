import json
import streamlit as st

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

from utils.llm_utils import refresh_db
from utils.llm_utils import create_llm, get_prompt
from utils.google_sheet_utils import (
    get_case_sheet_as_dict, 
    get_observation_sheet_as_dict,
    cases_related_to_observations,
    observations_related_to_cases
)

def fetch_similar_data(user_input):
    with st.chat_message("user"):
        st.markdown(user_input)

    # Perform similarity search using Pinecone
    updated_observations_db = refresh_db(namespace_to_refresh="observations")
    semantically_related_observations = updated_observations_db.similarity_search(user_input, k=3)

    cases_from_observations = cases_related_to_observations(semantically_related_observations)

    updated_cases_db = refresh_db(namespace_to_refresh="cases")
    semantically_related_cases = updated_cases_db.similarity_search(user_input, k=3)

    observations_from_cases = observations_related_to_cases(semantically_related_cases)

    return {"question": user_input, 
            "semantically_related_observations": semantically_related_observations,
            "cases_from_observations": cases_from_observations,
            "semantically_related_cases": semantically_related_cases,
            "observations_from_cases": observations_from_cases}

def fetch_real_time_gsheets_data(user_input):
    with st.chat_message("user"):
        st.markdown(user_input)

    return {"question": user_input, 
            "semantically_related_observations": json.dumps(get_observation_sheet_as_dict()),
            "semantically_related_cases": json.dumps(get_case_sheet_as_dict()),
            "cases_from_observations": "None",
            "observations_from_cases": "None"
            }


def get_retreiver_chain(vector_store):
  
  llm=create_llm()
  retriever = vector_store.as_retriever()
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
  ])
  history_retriver_chain = create_history_aware_retriever(llm, retriever, prompt)
  
  return history_retriver_chain


def get_conversational_rag(history_retriever_chain):
  llm=create_llm()

  all_observation_data = str(get_observation_sheet_as_dict())

  answer_prompt=ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:{observations}+\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}")
  ])

  document_chain = create_stuff_documents_chain(llm, answer_prompt)

  #create final retrieval chain
  conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain,document_chain)

  return conversational_retrieval_chain


def get_chat_response(user_input):

    llm = create_llm()

    updated_observations_db = refresh_db(namespace_to_refresh="observations")

    observations_retriever_chain = get_retreiver_chain(updated_observations_db)
    conversation_rag_chain = get_conversational_rag(observations_retriever_chain)

    # full_prompt = ChatPromptTemplate.from_messages(
    #     st.session_state.messages
    #     )
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.messages,
        "input": user_input,
        "observations": get_observation_sheet_as_dict()
    })

    return response["answer"]


def update_session(output):
    # Update the conversation history
    # st.session_state.messages.append({"role": "assistant", "content": output})
    # st.write(st.session_state.messages)

    # Display the response
    with st.chat_message("assistant"):
        st.markdown(output)

    # Store chat in the current sheet
    st.session_state.chat_sheet.append_row([st.session_state.messages[-2].content, st.session_state.messages[-1].content])
