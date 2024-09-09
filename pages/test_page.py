import streamlit as st

# Function to clear form inputs
def clear_form():
    st.session_state.need_statement = ''

# Function to handle form submission
def submit_form():
    # You can add any form submission logic here
    st.write("Text 1:", st.session_state.need_statement)
    # st.text_input(label="There is a need for...", st.session_state.need_statement)
    # Form submission logic
    if need_input:
            #st.session_state["need_statement"] = st.session_state["need_input"]
            # need_statement = need_input
            problem = problem_input
            population = population_input
            outcome = outcome_input
            notes = notes_input
            # update_need_ID()
            st.write("Need statement recorded!")
            # st.write(f'Relevant Observations: {observation_ID}')
            # st.write(f'Need ID: {st.session_state['need_ID']}')
            # st.write(f'Problem: {problem}')
            # st.write(f'Population: {population}')
            # st.write(f'Outcome: {outcome}')
            # st.write(f'Notes: {notes}')
            recordNeed(st.session_state['need_ID'], st.session_state['need_date'], st.session_state['need_statement'], problem, population, outcome, observation_ID, notes)
            # Clear the form after submission
            clear_form()
    

# Initialize the session state for the input if it doesn't exist
if 'need_statement' not in st.session_state:
    st.session_state.need_statement = ''

# Create the form
with st.form("my_form"):
    # Text input tied to session state
    st.text_input("Enter text 1", key='need_statement')
        #st.text_input("There is a need for...", value=st.session_state.need_statement)


    # Form submit button with a callback function
    submitted = st.form_submit_button("Submit", on_click=submit_form)

#
