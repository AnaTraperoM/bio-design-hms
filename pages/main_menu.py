import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from pydantic import BaseModel, Field
from typing import Optional
import csv
import os

# observations_csv = "observations.csv"

# class ObservationRecord(BaseModel):
#     date: Optional[str] = Field(default=None, description="Date of the observation")
#     location: Optional[str] = Field(default=None, description="Location or setting where this observation made. e.g. operating room (OR), hospital, exam room,....")
#     people_present: Optional[str] = Field(default=None, description="People present during the observation. e.g. patient, clinician, scrub tech, family member, ...")
#     sensory_observations: Optional[str] = Field(default=None, description="What is the observer sensing with sight, smell, sound, touch. e.g. sights, noises, textures, scents, ...")
#     specific_facts: Optional[str] = Field(default=None, description="The facts noted in the observation. e.g. the wound was 8cm, the sclera had a perforation, the patient was geriatric, ...")
#     insider_language: Optional[str] = Field(default=None, description="Terminology used that is specific to this medical practice or procedure. e.g. specific words or phrases ...")
#     process_actions: Optional[str] = Field(default=None, description="Which actions occurred during the observation, and when they occurred. e.g. timing of various steps of a process, including actions performed by doctors, staff, or patients, could include the steps needed to open or close a wound, ...")
#     summary_of_observation: Optional[str] = Field(default=None, description="A summary of 1 sentence of the encounter or observation, e.g. A rhinoplasty included portions that were functional (covered by insurance), and cosmetic portions which were not covered by insurance. During the surgery, the surgeon had to provide instructions to a nurse to switch between functional and cosmetic parts, back and forth. It was mentioned that coding was very complicated for this procedure, and for other procedures, because there are 3 entities in MEE coding the same procedure without speaking to each other, ...")
#     questions: Optional[str] = Field(default=None, description="Recorded open questions about people or their behaviors to be investigated later. e.g. Why is this procedure performed this way?, Why is the doctor standing in this position?, Why is this specific tool used for this step of the procedure? ...")

# if not os.path.exists(observations_csv):
#         observation_keys = list(ObservationRecord.__fields__.keys())
#         observation_keys = ['observation_title', 'observer', 'observation', 'observation_date'] + observation_keys        
#         csv_file = open(observations_csv, "w")
#         csv_writer = csv.writer(csv_file, delimiter=";")
#         csv_writer.writerow(observation_keys)


#current path to logo : Screenshot 2024-08-16 at 18.51.28.png

# Your logo URL
logo_url = "https://github.com/Aks-Dmv/bio-design-hms/blob/main/Screenshot%202024-08-16%20at%2018.51.28.png"  # Replace with the actual URL of your logo

# Display the title with the logo
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <h1 style='margin-right: 10px;'>HealthTech WayFinder</h1>
        <img src="{logo_url}" alt="Logo" style="width:50px; height:50px;">
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h3 style='text-align: center;'>What would you like to do?</h3>", unsafe_allow_html=True)


# # def main():
# st.markdown("<h1 style='text-align: center;'>HealthTech WayFinder</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center;'>What would you like to do?</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col2:
    if st.button("🔍 Record a New Observation"):
        switch_page("add_new_observation")

    if st.button("✅ Tips for your Observations"):
        switch_page("enhance_observation")

    if st.button("❓ Ask the team's Observations"):
        switch_page("ask_observations")

    if st.button("📊 Glossary"):
        switch_page("glossary")

    if st.button("📊 View All Observations"):
        switch_page("view_dataset")

    st.markdown("---")
    if st.button("Log Out"):
            switch_page("../app.py")
