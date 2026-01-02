#!/bin/bash

# TestGorilla AI Engineer Practice - Quick Start Script

echo "🚀 Starting TestGorilla AI Engineer Practice App..."
echo "=================================================="
echo ""

# Kill any existing Streamlit processes on port 8501
echo "Checking for existing processes on port 8501..."
lsof -ti:8501 | xargs kill -9 2>/dev/null
sleep 1

# Activate virtual environment
source ai_prep_env/bin/activate

echo "Launching application..."
echo "Access the app at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

# Start Streamlit app
streamlit run practice_app.py --server.port 8501 --server.headless true

echo ""
echo "Application stopped."
