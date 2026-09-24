#!/usr/bin/env bash
# Run the EW Smart Scan live demo dashboard.
# Usage: bash demo/run_demo.sh

set -e

# Check that streamlit is installed
python3 -c "import streamlit" 2>/dev/null || {
    echo "ERROR: streamlit is not installed."
    echo "Run: pip install -r requirements.txt"
    exit 1
}

echo "Starting EW Smart Scan Demo..."
echo "Open http://localhost:8501 in your browser."
echo ""
streamlit run demo/app.py
