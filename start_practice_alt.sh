#!/bin/bash

# TestGorilla AI Engineer Practice - Alternative Port Script

echo "🚀 Starting TestGorilla AI Engineer Practice App on alternative port..."
echo "========================================================================"
echo ""

# Verify topic resources exist
if [ -f "topic_resources.json" ] && [ -f "must_know_topics.json" ]; then
    echo "✅ Topic resources loaded"
    TOPIC_COUNT=$(python3 -c "import json; f=open('topic_resources.json'); d=json.load(f); print(len(d.get('topics', {})))" 2>/dev/null || echo "?")
    echo "   - $TOPIC_COUNT topics with learning resources available"
    echo "   - 10 categories with must-know topic lists"
else
    echo "⚠️  Topic resources not found (optional feature)"
fi
echo ""

# Use alternative port 8502
PORT=8502

# Activate virtual environment
source ai_prep_env/bin/activate

echo "Launching application on port $PORT..."
echo "========================================================================"
echo "Access the app at: http://localhost:$PORT"
echo ""
echo "Features:"
echo "  📝 320 practice questions across 30+ categories"
echo "  📚 Progressive learning resources (NEW!)"
echo "  📖 Must-know topics for each domain"
echo "  📊 Progress tracking and statistics"
echo ""
echo "Navigation:"
echo "  • Click '📚 Learn Topics' for learning resources"
echo "  • Click '📖 Must-Know' for essential topic lists"
echo "  • Answer questions and click '📚 Learn Topic' for deep dives"
echo ""
echo "Press Ctrl+C to stop the application"
echo "========================================================================"
echo ""

# Start Streamlit app on alternative port
streamlit run practice_app.py --server.port $PORT --server.headless true

echo ""
echo "Application stopped."
