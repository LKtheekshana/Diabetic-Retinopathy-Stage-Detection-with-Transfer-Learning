@echo off
REM Launch the Diabetic Retinopathy Screening Assistant in the browser.
cd /d "%~dp0"
streamlit run app.py
pause
