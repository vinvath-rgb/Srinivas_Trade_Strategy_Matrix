import streamlit as st

st.set_page_config(page_title="Strategy–Regime Matrix (Srini)", layout="wide")

from regime_matrix_app.streamlit_regime_app import main as app_main

def main():
    app_main()

if __name__ == "__main__":
    main()
