import streamlit as st
from util import session as ss

def authenticated_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("pages/page_dashboard_interest.py", label="관심종목")
    st.sidebar.divider()
    st.sidebar.page_link("pages/page_list_searched.py", label="관심등록")
    st.sidebar.page_link("pages/page_chart_analysis_basic.py", label="차트분석")
    st.sidebar.divider()
    logout = st.sidebar.button(label="Logout")
    if logout:
        ss.clear_session_logout()
    #if st.session_state.role in ["admin"]:
    #    st.sidebar.page_link("pages/page2.py", label="차트분석")

def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
#    if "role" not in st.session_state:
#        st.switch_page("login.py")
    authenticated_menu()
