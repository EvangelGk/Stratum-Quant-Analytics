@echo off
setlocal
cd /d %~dp0

echo Starting Quant Real-Time Lab...
streamlit run UI/realtime_lab.py

endlocal
