#!/bin/bash
pip install -r /content/requirements.txt
npm install localtunnel
streamlit run /content/ex-2.py &>/content/logs.txt &

# Wait for Streamlit to start up
sleep 3

npx localtunnel --port 8501
