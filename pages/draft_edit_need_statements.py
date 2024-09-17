# ////////////////////// IMPORTS ////////////////////// IMPORTS ////////////////////// IMPORTS //////////////////////
import time
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from datetime import date
import logging
logging.basicConfig(level=logging.INFO)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
# from langchain.callbacks import get_openai_callback
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore

import gspread
from oauth2client.service_account import ServiceAccountCredentials


from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime

import json
import os
import csv

# ////////////////////// INITIALIZATIONS ////////////////////// INITIALIZATIONS ////////////////////// INITIALIZATIONS ////////////////////// 

OPENAI_API_KEY = st.secrets["openai_key"]

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

# Initialize the session state for the input if it doesn't exist
if 'need_ID_with_preview' not in st.session_state:
    st.session_state.need_ID_with_preview = ''

if 'need_statement' not in st.session_state:
    st.session_state.need_statement = ''

if 'problem' not in st.session_state:
    st.session_state['problem'] = ""

if 'population' not in st.session_state:
    st.session_state['population'] = ""

if 'outcome' not in st.session_state:
    st.session_state['outcome'] = ""

if 'notes' not in st.session_state:
    st.session_state['notes'] = ""

if 'observation_ID' not in st.session_state:
    st.session_state['observation_ID'] = ""

if 'result' not in st.session_state:
    st.session_state['result'] = ""

if 'rerun' not in st.session_state:
    st.session_state['rerun'] = False

if 'selected_need_ID' not in st.session_state:
    st.session_state['selected_need_ID'] = ""

# ////////////////////// FUNCTIONS ////////////////////// FUNCTIONS ////////////////////// FUNCTIONS ////////////////////// 


# get need IDs with preview
def getExistingNeedIDS():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.metadata.readonly"
        ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    need_log = client.open("2024 Healthtech Identify Log").worksheet("Need Statement Log")
    need_ids = need_log.col_values(1)[1:]
    need_previews = need_log.col_values(3)[1:]

    # find all observation ids with the same date
    existing_need_ids_with_title = dict(zip(need_ids, need_previews))

    # make strings with case id - title
    existing_need_ids_with_title = [f"{need_ids} - {need_previews}" for need_ids, need_previews in existing_need_ids_with_title.items()]

    print("Existing Observation IDS: ")
    print(existing_need_ids_with_title)
    return existing_need_ids_with_title


# Fetch case details based on selected case ID
def fetch_need_details(selected_need_ID):
    sheet = get_google_sheet("2024 Healthtech Identify Log", "Need Statement Log")
    need_data = sheet.get_all_records()

    # # Print the need_data being fetched
    # st.write(need_data)

    for row in need_data:
        if "need_ID" in row and row["need_ID"].strip() == need_id.strip():
            return row
    
    st.error(f"Need ID {need_id} not found.")
    return None


# Update case details in Google Sheets
def update_need(selected_need_ID, updated_need_data):
    # i = 0
    try:
        sheet = get_google_sheet("2024 Healthtech Identify Log", "Need Log")
        data = sheet.get_all_records()

        # Find the row corresponding to the selected_need_ID and update it
        for i, row in enumerate(data, start=2):  # Skip header row
            if row["need_ID"] == selected_need_ID:
                # Update the necessary fields (Assuming the updated_need_data has the same keys as Google Sheets columns)
                for key, value in updated_need_data.items():
                    sheet.update_cell(i, list(row.keys()).index(key) + 1, value)
                return True
        return False
    except Exception as e:
        print(f"Error updating case: {e}")
        return False
    
# ////////////////////// CODE ON PAGE ////////////////////// CODE ON PAGE ////////////////////// CODE ON PAGE //////////////////////


st.markdown("### Edit a Need Statement")



# Dropdown menu for selecting action
# action = st.selectbox("Choose an action", ["Add New Case", "Edit Existing Case"])


# select from a list of needs
existing_need_ids_with_title = getExistingNeedIDS()
st.session_state['need_ID_with_preview'] = st.selectbox("Select Need Statement", existing_need_ids_with_title)

# get ID from the dropdown value
st.session_state['selected_need_ID'] = st.session_state.need_ID_with_preview.split(" - ")[0]

if selected_need_ID: #may need to make this session state whatever
    need_details = fetch_need_details(need_to_edit)


    # need_details = fetch_need_details(need_to_edit)
    if need_details:
        # # Debug: Print the case details (optional)
        # st.write(f"Editing case: {need_details}")
        # Editable fields for the selected case
        case_title = st.text_input("Title", need_details.get("Title", ""))
        #case_date = st.date_input("Date", date.fromisoformat(need_details.get("Date", str(date.today()))))
        case_description = st.text_area("Case Description", need_details.get("Case Description", ""))
        location = st.text_input("Location", need_details.get("Location", ""))
        stakeholders = st.text_input("Stakeholders", need_details.get("Stakeholders", ""))
        people_present = st.text_input("People Present", need_details.get("People Present", ""))
        insider_language = st.text_input("Insider Language", need_details.get("Insider Language", ""))
        tags = st.text_input("Tags", need_details.get("Tags", ""))
        observations = st.text_area("Observations", need_details.get("Observations", ""))

         # Get and validate the date field
        case_date_str = need_details.get("Date", "")
        try:
                    # Try to parse the date from ISO format, or default to today's date
            case_date = date.fromisoformat(case_date_str) if case_date_str else date.today()
        except ValueError:
            case_date = date.today()

        case_date_input = st.date_input("Date", case_date)
            
                # Step 3: Save changes
        if st.button("Save Changes"):
            updated_need_data = {
                "Title": case_title,
                "Date": case_date_input.isoformat(),
                "Case Description": case_description,
                "Location": location,
                "Stakeholders": stakeholders,
                "People Present": people_present,
                "Insider Language": insider_language,
                "Tags": tags,
                "Observations": observations,
            }
            
            if update_need(need_to_edit, updated_need_data):
                st.success(f"Changes to '{need_to_edit}' saved successfully!")
            else:
                st.error(f"Failed to save changes to '{need_to_edit}'.")







# ////////////////////// NOTES ////////////////////// NOTES ////////////////////// NOTES ////////////////////// 













