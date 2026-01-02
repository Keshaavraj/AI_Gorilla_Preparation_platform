#!/bin/bash

# TestGorilla AI Engineer Practice - Alternative Port Script

echo "🚀 Starting TestGorilla AI Engineer Practice App on alternative port..."
echo "========================================================================"
echo ""

# Use alternative port 8502
PORT=8502

# Activate virtual environment
source ai_prep_env/bin/activate

echo "Launching application on port $PORT..."
echo "Access the app at: http://localhost:$PORT"
echo "Press Ctrl+C to stop the application"
echo ""

# Start Streamlit app on alternative port
streamlit run practice_app.py --server.port $PORT --server.headless true

echo ""
echo "Application stopped."
