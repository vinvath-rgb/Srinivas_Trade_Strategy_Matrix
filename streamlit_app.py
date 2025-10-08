# streamlit_app.py
import os
import streamlit as st

# MUST be the first Streamlit call in the whole app
st.set_page_config(page_title="Strategy–Regime Matrix (Srini)", layout="wide")

from regime_matrix_app.strategy_regime_matrix_app import main as app_main
# import AFTER set_page_config
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