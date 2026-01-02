# Port Issue Fix Guide

## Problem: Port 8501 Already in Use

If you see the error: `Port 8501 is already in use`

## Solutions:

### Option 1: Use Updated Start Script (Recommended)
The start script now automatically kills any existing process on port 8501:

```bash
./start_practice.sh
```

### Option 2: Use Alternative Port
Start the app on port 8502 instead:

```bash
./start_practice_alt.sh
```

Then access at: `http://localhost:8502`

### Option 3: Manual Port Kill
Manually kill the process and restart:

```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Wait a moment
sleep 1

# Restart
./start_practice.sh
```

### Option 4: Find and Kill Streamlit Processes
```bash
# Find all Streamlit processes
ps aux | grep streamlit

# Kill all Streamlit processes
pkill -f streamlit

# Restart
./start_practice.sh
```

### Option 5: Start Manually on Custom Port
```bash
source ai_prep_env/bin/activate
streamlit run practice_app.py --server.port 8503
```

## Checking if App is Running

```bash
# Check if port 8501 is in use
lsof -i:8501

# Check all Streamlit processes
ps aux | grep streamlit
```

## Stopping the App

When the app is running, press **Ctrl+C** in the terminal to stop it gracefully.

---

**The updated start script should handle this automatically now!**
