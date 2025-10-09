import os
import streamlit as st

st.set_page_config(page_title="Strategy–Regime Matrix (Srini)", layout="wide")

from regime_matrix_app.streamlit_regime_app import main as app_main


def _auth():
    pw = os.getenv("APP_PASSWORD", "")
    if not pw:
        return True
    with st.sidebar:
        st.subheader("🔐 App Login")
        inp = st.text_input("Password", type="password")
    if inp != pw:
        st.stop()
    return True


def main():
    _auth()
    app_main()


if __name__ == "__main__":
    main()
    