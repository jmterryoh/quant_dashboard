# pip install streamlit==1.32.0
import streamlit as st
from util import screen as sc
from db import db_client as dc
from util import session as ss

st.set_page_config(initial_sidebar_state="collapsed")
sc.hide_sidebar()

# Streamlit app
def main():

    st.title("Login")

    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if password or login_button:
        task_name = "authenticate"
        params = {'username': f"{username}", 'password': f"{password}"}
        respose = dc.fetch_result_from_remote_server(task_name, params)
        if "return" in respose:
            if "result" in respose["return"]:
                if respose["return"]["result"] == "success":
                    role = respose["return"]["data"]["role"]
                    ss.create_session(role=role)
                    st.success("Login successful!")
                    # Redirect to page1.py with session token
                    st.switch_page("pages/page_list_searched.py")
                else:
                    st.error("Invalid username or password.")
            else:
                st.error("Invalid username or password.")
        else:
            st.error("Invalid username or password.")

if __name__ == "__main__":
    main()
