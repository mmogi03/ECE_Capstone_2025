#!/bin/bash
# kill_services.sh
# This script kills all processes associated with our application:
# - main.py (which launches the DL inference and WebSocket server)
# - dl_inference.py (if spawned separately)
# - ngrok (the tunnel process)
# - any process listening on port 8765

echo "Killing main.py processes (and any of its children)..."
pkill -f main.py

echo "Killing any processes referencing dl_inference.py..."
pkill -f dl_inference.py

echo "Killing ngrok processes..."
pkill -f ngrok

# Check if port 8765 is in use and kill any process using it.
if lsof -Pi :8765 -sTCP:LISTEN -t >/dev/null; then
    echo "Port 8765 is in use. Killing process(es)..."
    kill -9 $(lsof -t -i:8765)
fi

echo "All specified processes have been terminated."

