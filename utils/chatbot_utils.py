import json
import os
from typing import List
import streamlit as st

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool


from utils.chatbot_parameters import SYSTEM_PROMPT
from utils.llm_utils import refresh_db
from utils.llm_utils import create_llm, get_prompt
from utils.google_sheet_utils import (
    get_case_sheet_as_dict, 
    get_observation_sheet_as_dict,
    cases_related_to_observations,
    observations_related_to_cases,
    get_need_statement_sheet_as_dict
)

def fetch_similar_data(user_input):

    # Perform similarity search using Pinecone
    updated_observations_db = refresh_db(namespace_to_refresh="observations_temp")
    semantically_related_observations = updated_observations_db.similarity_search(user_input, k=20)

    cases_from_observations = cases_related_to_observations(semantically_related_observations)

    updated_cases_db = refresh_db(namespace_to_refresh="cases_temp")
    semantically_related_cases = updated_cases_db.similarity_search(user_input, k=20)

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

@tool
def get_observations_from_need_statements(list_of_need_statement_ids: List[str]) -> str:
    '''
    Gets the observations linked to the need statements

    Args:
    list_of_need_statement_ids: List of need statement IDs

    Returns:
    str: Observations linked to the need statements

    '''

    assert type(list_of_need_statement_ids) == list, "list_of_need_statement_ids should be a list"

    need_statements_in_sheet = get_need_statement_sheet_as_dict()
    observations_in_sheet = get_observation_sheet_as_dict()

    need_statements_by_id = {need_statement["need_ID"]: need_statement for need_statement in need_statements_in_sheet.values()}
    observations_by_id = {observation["Observation ID"]: observation for observation in observations_in_sheet.values()}

    observations = []
    for need_statement_id in list_of_need_statement_ids:
        
        linked_observation_id = need_statements_by_id[need_statement_id]["observation_ID"]
        observations.append(observations_by_id[linked_observation_id])

    return """
    Observations linked to the need statements are: {observations}
"""


def create_chatbot_chain():
    llm=create_llm()

    observation_retriever = refresh_db(namespace_to_refresh=st.session_state.observation_namespace).as_retriever(search_kwargs={'k': 20})

    doc_prompt = PromptTemplate.from_template(
    """Observation ID: {Observation ID}
Description: {page_content}
Observer: {Observer}
Date: {Date}"""
    )

    observation_retriever_tool = create_retriever_tool(
        observation_retriever,
        name="observations_retriever",
        description="Searches and returns clinical observations related to the user query.",
        document_prompt=doc_prompt,

    )

    need_statement_retriever = refresh_db(namespace_to_refresh=st.session_state.need_statement_namespace).as_retriever(search_kwargs={'k': 10})

    need_statement_doc_prompt = PromptTemplate.from_template(
    """Need ID: {need_ID}
Observation ID: {observation_ID}
Statement: {page_content}
"""
    )

    need_statement_retriever_tool = create_retriever_tool(
        need_statement_retriever,
        name="need_statement_retriever",
        description="Searches and returns need statements related to the user query. If this is called then the get_observations_from_need_statements has to be called next.",
        document_prompt=need_statement_doc_prompt,
    )

    # os.environ["SERPER_API_KEY"] = "1f5fbd41f519e591303f78cb58caf9794ba43dc7"
    # search = GoogleSerperAPIWrapper()

    # search_tool = GoogleSerperAPIWrapper(
    #     search.run,
    #     name="Search for answers on internet",
    #     description="useful for when you need to ask with search",
    # )

    tools = [observation_retriever_tool, need_statement_retriever_tool, get_observations_from_need_statements]

    memory_saver = MemorySaver()

    agent_executor = create_react_agent(llm, tools, checkpointer=memory_saver, state_modifier=SystemMessage(content=SYSTEM_PROMPT))

    return agent_executor


def get_chat_response(user_input):
    if 'chatbot_chain' not in st.session_state:
            st.session_state.chatbot_chain = create_chatbot_chain()
            st.session_state.chatbot_config = {"configurable": {"thread_id": "abc123"}}

    response =  ""
    
    for s in st.session_state.chatbot_chain.stream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="values",
            config=st.session_state.chatbot_config
        ):
        response_message = s['messages'][-1]
        response = response_message.content

        st.write(response_message.pretty_print())

    # return response['agent']['messages'][0].content
    return response



# def create_chatbot_chain():
#     llm=create_llm()

#     answer_prompt=ChatPromptTemplate.from_messages([
#         SystemMessage(content=SYSTEM_PROMPT),
#         ("assistant", "I have found the following observations: {observations} and cases: {cases} relevant"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}")
#     ])

#     chatbot_chain = answer_prompt | llm | StrOutputParser()

#     return chatbot_chain

# def get_chat_response(user_input):

#     if 'chatbot_chain' not in st.session_state:
#         st.session_state.chatbot_chain = create_chatbot_chain()

#     return st.session_state.chatbot_chain.stream({
#         "chat_history": st.session_state.messages,
#         "input": user_input,
#         "observations": get_observation_sheet_as_dict(),
#         "cases": get_case_sheet_as_dict()
#     })


def update_session(output):
    # Update the conversation history
    # st.session_state.messages.append({"role": "assistant", "content": output})
    # st.write(st.session_state.messages)

    # # Display the response
    # with st.chat_message("assistant"):
    #     st.markdown(output)

    # Store chat in the current sheet
    st.session_state.chat_sheet.append_row([st.session_state.messages[-2].content, st.session_state.messages[-1].content])
