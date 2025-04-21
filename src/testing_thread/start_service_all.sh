#!/bin/bash
# run_services.sh
# This script checks if port 8765 is in use, kills offending processes,
# then launches the main.py process (which launches both DL inference and the WebSocket server)
# and starts an ngrok tunnel on port 8765.

# Check if port 8765 is in use and kill any process using it.
if lsof -Pi :8765 -sTCP:LISTEN -t >/dev/null; then
    echo "Port 8765 is in use. Killing process(es)..."
    kill -9 $(lsof -t -i:8765)
    sleep 1
fi

# Trap SIGINT and SIGTERM to kill child processes on exit.
trap "echo 'Stopping services...'; kill $MAIN_PID $NGROK_PID; exit 0" SIGINT SIGTERM

echo "Starting main Python process (main.py)..."
python3 main.py &
MAIN_PID=$!

# Give the main process a moment to initialize.
sleep 2

echo "Starting ngrok tunnel on port 8765..."
ngrok http 8765 --host-header=localhost:8765 &
NGROK_PID=$!

echo "Services started:"
echo " - Main process PID: $MAIN_PID"
echo " - ngrok PID: $NGROK_PID"
echo "ngrok Web Interface: http://127.0.0.1:4040"

# Wait indefinitely.
wait
