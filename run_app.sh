#!/bin/bash
pip install -r requirements.txt
npm install localtunnel
streamlit run langchain_based_app.py &>/logs.txt &

# Wait for Streamlit to start up
sleep 3

npx localtunnel --port 8501
