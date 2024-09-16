import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_cookies_manager import CookieManager  # Correct import
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests

# Initialize cookies manager
cookies = CookieManager()

# OAuth setup: Using Streamlit secrets instead of client_secret.json
def get_google_oauth_flow():
    client_config = {
        "web": {
            "client_id": st.secrets["google_oauth"]["client_id"],
            "project_id": st.secrets["google_oauth"]["project_id"],
            "auth_uri": st.secrets["google_oauth"]["auth_uri"],
            "token_uri": st.secrets["google_oauth"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["google_oauth"]["auth_provider_x509_cert_url"],
            "client_secret": st.secrets["google_oauth"]["client_secret"],
            "redirect_uris": ["https://healthtech-wayfinder-playground.streamlit.app"]  # Use base URL
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"]
    )
    
    flow.redirect_uri = st.secrets["google_oauth"]["redirect_uris"][0]
    
    return flow

# Get Google Authorization URL
def initiate_google_flow():
    flow = get_google_oauth_flow()
    auth_url, state = flow.authorization_url(prompt='consent')
    
    st.session_state['oauth_state'] = state  # Save state for validation later
    return auth_url

# Exchange the authorization code for credentials and verify email
def exchange_code_for_credentials(flow, code):
    flow.fetch_token(code=code)
    credentials = flow.credentials

    try:
        id_info = id_token.verify_oauth2_token(
            credentials.id_token, requests.Request(), st.secrets["google_oauth"]["client_id"]
        )
    except ValueError:
        st.error("Invalid Google OAuth token. Please try again.")
        return None

    user_email = id_info.get('email', None)
    if not user_email:
        st.error("Could not retrieve email from Google account.")
        return None

    return user_email

# Function to hide the sidebar (if needed)
def hide_sidebar():
    hide_sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Check if the user selected "Stay logged in" and is already logged in via cookies
def check_stay_logged_in():
    # Attempt to load cookies and handle any errors
    if cookies.ready():
        cookies.load()  # Load the cookies into memory
        
        if "login_status" not in st.session_state:
            st.session_state["login_status"] = "not_logged_in"

        # Check the cookies to see if the user is logged in
        if cookies.get("logged_in") == "true":
            st.session_state["login_status"] = "success"
            st.session_state["google_user"] = cookies.get("google_user")
    else:
        st.warning("Cookies are not ready, unable to check login state.")

# Main login function
def main():
    st.markdown("<h2 style='text-align: center;'>Welcome back! </h2>", unsafe_allow_html=True)
    
    logo_url = "https://raw.githubusercontent.com/Aks-Dmv/bio-design-hms/main/Logo-HealthTech.png"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1>HealthTech Wayfinder</h1>
            <img src="{logo_url}" alt="Logo" style="width:350px; height:auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Login form
    with st.form(key="login_form"):
        st.write("Login:")
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        stay_logged_in = st.checkbox("Stay logged in")
        submit_button = st.form_submit_button("Submit")
    
    if submit_button:
        user_list = st.secrets["login-credentials"]
        for user_dict in user_list:
            if username == user_dict["username"] and password == user_dict["password"]:
                st.success("Login successful")
                st.session_state["login_status"] = "success"
                
                # Set the "stay logged in" cookie if checked
                if stay_logged_in:
                    cookies["logged_in"] = "true"
                    cookies["google_user"] = username  # Save user info
                else:
                    cookies["logged_in"] = "false"
                    cookies["google_user"] = ""

                cookies.save()  # Save cookies to the browser
                return
        else:
            st.error("Invalid username or password")

    # Google Login
    st.write("Or use Google to log in:")
    
    # Generate the OAuth URL
    auth_url = initiate_google_flow()
    
    st.markdown(f'<a href="{auth_url}" target="_self">Click here to log in with Google</a>', unsafe_allow_html=True)

    # Process Google authentication callback
    query_params = st.experimental_get_query_params()
    if 'code' in query_params and 'oauth_state' in st.session_state:
        flow = get_google_oauth_flow()

        if query_params.get('state', [''])[0] == st.session_state['oauth_state']:
            user_email = exchange_code_for_credentials(flow, query_params['code'][0])

            if user_email:
                allowed_emails = st.secrets["allowed_emails"]["emails"]

                if user_email in allowed_emails:
                    st.session_state["login_status"] = "success"
                    st.session_state["google_user"] = user_email  # Store the email for later use
                    
                    cookies["logged_in"] = "true"
                    cookies["google_user"] = user_email
                    cookies.save()

                    st.success(f"Google login successful for {user_email}!")
                    switch_page("Dashboard")
                    return
                else:
                    st.error("Unauthorized email. Access denied.")

# Main app logic
if __name__ == "__main__":
    # Check if the user chose to stay logged in
    check_stay_logged_in()

    if st.session_state.get("login_status") == "success":
        st.sidebar.write(f"You are logged in as {st.session_state['google_user']}!")
        st.sidebar.write("You can access the menu.")
        switch_page("Dashboard")
    else:
        hide_sidebar()
        main()
