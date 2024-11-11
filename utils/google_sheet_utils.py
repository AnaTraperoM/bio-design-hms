import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.documents import Document


# Access the credentials from Streamlit secrets
creds_dict = {
    "type" : st.secrets["gwf_service_account"]["type"],
    "project_id" : st.secrets["gwf_service_account"]["project_id"],
    "private_key_id" : st.secrets["gwf_service_account"]["private_key_id"],
    "private_key" : st.secrets["gwf_service_account"]["private_key"],
    "client_email" : st.secrets["gwf_service_account"]["client_email"],
    "client_id" : st.secrets["gwf_service_account"]["client_id"],
    "auth_uri" : st.secrets["gwf_service_account"]["auth_uri"],
    "token_uri" : st.secrets["gwf_service_account"]["token_uri"],
    "auth_provider_x509_cert_url" : st.secrets["gwf_service_account"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url" : st.secrets["gwf_service_account"]["client_x509_cert_url"],
    "universe_domain": st.secrets["gwf_service_account"]["universe_domain"],
}


# Google Sheets setup
SCOPE = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.metadata.readonly"
        ]
CREDS = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
CLIENT = gspread.authorize(CREDS)
SPREADSHEET = CLIENT.open("Observation Investigator - Chat Log")  # Open the main spreadsheet

def create_new_chat_sheet():
    """Create a new sheet for the current chat thread."""
    chat_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Unique name based on timestamp
    sheet = SPREADSHEET.add_worksheet(title=f"Chat_{chat_timestamp}", rows="1", cols="2")  # Create new sheet
    sheet.append_row(["User Input", "Assistant Response"])  # Optional: Add headers
    return sheet

# Deprecated function
# def get_observation_sheet_as_dict():
#     observation_sheet = CLIENT.open("2024 Healthtech Identify Log").worksheet("Observation Log")
#     data = observation_sheet.get_all_records()
#     return data

def get_case_sheet_as_dict():
    case_sheet = CLIENT.open("Copy of 2024 Healthtech Identify Log").worksheet("Case Log")
    data = case_sheet.get_all_records()
    return data

def get_observation_sheet_as_dict():
    observation_sheet = CLIENT.open("Copy of 2024 Healthtech Identify Log").worksheet("Observation Log")
    data = observation_sheet.get_all_records()
    return data

def get_case_descriptions_from_case_ids(case_ids):
    data = get_case_sheet_as_dict()

    return {
        case['Case ID']: case['Case Description']
        for case in data
        if case['Case ID'] in case_ids
    }

def get_observation_descriptions_from_observation_ids(observation_ids):
    data = get_observation_sheet_as_dict()

    return {
        observation['Observation ID']: observation['Observation Description']
        for observation in data
        if observation['Observation ID'] in observation_ids
    }

def cases_related_to_observations(list_of_observations_pinecone):
    case_ids = []
    for observation in list_of_observations_pinecone:
        case_ids.append(observation.metadata['Case ID'])
    return get_case_descriptions_from_case_ids(case_ids)

def observations_related_to_cases(list_of_cases_pinecone):
    observation_ids = []
    for case in list_of_cases_pinecone:
        observation_ids.append(case.metadata['Observation ID'])
    return get_observation_descriptions_from_observation_ids(observation_ids)


def sync_with_pinecone(namespace='temp'):
    """Sync the Google Sheet data with Pinecone"""
    # Get the data from the Google Sheets
    # cases_iterable = get_case_sheet_as_dict()
    observations_in_sheet = get_observation_sheet_as_dict()

    # Create a Pinecone vector store
    db = PineconeVectorStore(
        index_name=st.secrets["pinecone-keys"]["index_to_connect"],
        namespace=namespace,
        embedding=OpenAIEmbeddings(api_key=st.secrets["openai_key"]),
        pinecone_api_key=st.secrets["pinecone-keys"]["api_key"],
    )

    st.write("Syncing data with Pinecone...")

    start_time = datetime.now()

    pinecone_ids = list(db._index.list(namespace=namespace))[0]

    pinecone_data = db._index.fetch(ids=pinecone_ids, namespace=namespace)

    observation_ids = [observation['Observation ID'] for observation in observations_in_sheet]
    observation_descriptions = [observation['Observation Description'] for observation in observations_in_sheet]
    observation_metadatas = [{k: v for k, v in observation.items() if k not in ['Observation ID', 'Observation Description']} for observation in observations_in_sheet]

    observations_added = db.add_texts(observation_descriptions, metadatas=observation_metadatas, ids=observation_ids)


    # for observation in observations_in_sheet:
    #     observation_id = observation['Observation ID']
    #     observation_description = observation['Observation Description']

    #     # all other metadata as a dict
    #     metadata = {k: v for k, v in observation.items() if k not in ['Observation ID', 'Observation Description']}

    #     if observation_id not in pinecone_ids:
    #         db.add_texts([observation_description], metadatas=[metadata], ids=[observation_id])

    #     if observation_id in pinecone_ids:
    #         existing_observation_description = pinecone_data[observation_id].metadata['text']
    #         if existing_observation_description != observation_description:
    #             db.add_texts([observation_description], ids=[observation_id])
    
    st.write("Number of observations added: ", observations_added)

    st.write("Done syncing data with Pinecone in ", datetime.now() - start_time)

    return db