# streamlit_app.py
import os
import streamlit as st
from regime_matrix_app.Streamlit_Regime_App import main as app_main

st.set_page_config(page_title="Strategy–Regime Matrix (Srini)", layout="wide")

def _auth():
    pw_env = os.getenv("APP_PASSWORD", "")
    if not pw_env:
        return  # no auth configured
    with st.sidebar:
        st.subheader("🔐 App Login")
        pw = st.text_input("Password", type="password", key="auth_password")
    if pw != pw_env:
        st.stop()

def run():
    _auth()
    app_main()

if __name__ == "__main__":
    run()