import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate

from utils.chatbot_parameters import SYSTEM_PROMPT, LLM_MODEL_NAME, LLM_TEMP, LLM_TOKENS

OPENAI_API_KEY = st.secrets["openai_key"]


def refresh_db(namespace_to_refresh):
    db = PineconeVectorStore(
        index_name=st.secrets["pinecone-keys"]["index_to_connect"],
        namespace=namespace_to_refresh,
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        pinecone_api_key=st.secrets["pinecone-keys"]["api_key"],
    )
    return db


def create_llm():
    return ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMP,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=LLM_TOKENS,
    )

def get_prompt():
    question_prompt = PromptTemplate.from_template(
        SYSTEM_PROMPT +  """
        Question: {question}
        Semantically Relevant Observations: {semantically_related_observations}
        Relevant Cases linked to above Observations: {cases_from_observations}

        Semantically Relevant cases: {semantically_related_cases}
        Relevant Observations linked to above Cases: {observations_from_cases}

        Final Answer:
        """
    )

    return question_prompt