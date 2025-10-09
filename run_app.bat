@echo off
setlocal
cd /d "%~dp0"
echo launching Streamlit app...
py -m streamlit run "%~dp0streamlit_app.py"
pause