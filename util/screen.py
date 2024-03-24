import streamlit as st
from streamlit_js_eval import streamlit_js_eval

def hide_sidebar():
    st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        div[data-testid="stStatusWidget"]{
            display: none !important;
        }
        section[data-testid="stSidebar"]{
            min-width: 1px !important;
            width: 2px !important; # Set the width to your desired value
            display: none!important;
        }
    </style>
    """, unsafe_allow_html=True)

def show_min_sidebar():
    st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        div[data-testid="stStatusWidget"]{
            display: none !important;
        }
        section[data-testid="stSidebar"]{
            min-width: 80px !important;
            width: 140px !important; # Set the width to your desired value
            display: none;
        }
        .st-emotion-cache-z5fcl4 {padding-top:3rem!important;}                                      
    </style>
    """, unsafe_allow_html=True)

def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
    
