import json
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


from utils.chatbot_parameters import SYSTEM_PROMPT
from utils.llm_utils import refresh_db
from utils.llm_utils import create_llm, get_prompt
from utils.google_sheet_utils import (
    get_case_sheet_as_dict, 
    get_observation_sheet_as_dict,
    cases_related_to_observations,
    observations_related_to_cases
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


def create_chatbot_chain():
    llm=create_llm()

    retriever = refresh_db(namespace_to_refresh=st.session_state.observation_namespace).as_retriever(search_kwargs={'k': 20})

    doc_prompt = PromptTemplate.from_template(
    """Observation ID: {Observation ID}
Description: {page_content}
Observer: {Observer}
Date: {Date}"""
    )

    tool = create_retriever_tool(
        retriever,
        name="observations_retriever",
        description="Searches and returns clinical observations related to the user query.",
        document_prompt=doc_prompt,

    )
    tools = [tool]

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
            # stream_mode="values",
            config=st.session_state.chatbot_config
        ):
        response = s

    return response['agent']['messages'][0].content



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
