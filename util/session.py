import streamlit as st
import time

PAGE_LANDING = "login.py"
PAGE_TIMEOUT = 1800 #in seconds

# Function to check session expiration
def check_session(last_page):

    # Here, we assume session expires after PAGE_TIMEOUT minutes
    current_time = time.time()
    if 'logged_in' not in st.session_state or 'last_time' not in st.session_state:
        st.switch_page(PAGE_LANDING)
        
    session_timestamp = st.session_state['last_time']
    if current_time - session_timestamp > PAGE_TIMEOUT:  # PAGE_TIMEOUT minutes in seconds
        # If session has expired, delete session information from cache
        st.error("Session expired. Redirecting to main page...")
        clear_session_logout()
    else:
        # If session is still valid, update session timestamp in cache
        st.session_state['last_time'] = current_time
        st.session_state['last_page'] = last_page
        return True
    
def create_session(role='user'):
    st.session_state['logged_in'] = True
    st.session_state['last_time'] = time.time()
    st.session_state['role'] = role

def get_session_last_page():
    if 'logged_in' in st.session_state:
        if st.session_state['logged_in']:
            if 'last_page' in st.session_state:
                return st.session_state['last_page']
    return None

def clear_session_logout():
    if 'logged_in' in st.session_state:
        del st.session_state['logged_in']
    if 'last_time' in st.session_state:
        del st.session_state['last_time']
    if 'role' in st.session_state:
        del st.session_state['role']
    if 'last_page' in st.session_state:
        del st.session_state['last_page']
    for key in st.session_state.keys():
        del st.session_state[key]
    st.switch_page(PAGE_LANDING)


