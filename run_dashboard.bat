@echo off
title LK-PhonePrice: Ensemble Learning Dashboard
echo =====================================================================
echo  LK-PhonePrice: Ensemble Learning & Hybridization Dashboard
echo  Predicting Used & New Mobile Phone Prices in Sri Lanka (ikman.lk)
echo =====================================================================
echo.
echo Launching Streamlit Dashboard...
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Streamlit encountered an issue or is still installing packages.
    echo Trying direct streamlit command...
    streamlit run app.py
)
pause
