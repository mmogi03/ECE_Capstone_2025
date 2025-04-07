#!/bin/bash

# Check if port 8765 is in use and kill any process using it
if lsof -Pi :8765 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 8765 is in use. Killing process(es)..."
    kill -9 $(lsof -t -i:8765)
    sleep 1
fi

# Trap Ctrl+C (SIGINT) and SIGTERM to gracefully kill child processes
trap "echo 'Stopping services...'; kill $MAIN_PID $NGROK_PID; exit 0" SIGINT SIGTERM

echo "Starting main Python process (server + DL inference)..."
# Start the main script that launches the server and DL inference as separate processes.
python3 main.py &
MAIN_PID=$!

# Give the main process a moment to initialize.
sleep 2

echo "Starting ngrok tunnel on port 8765..."
# Start ngrok with the host header option.
ngrok http 8765 --host-header=localhost:8765 &
NGROK_PID=$!

echo "Services started:"
echo " - Main process PID: $MAIN_PID"
echo " - ngrok PID: $NGROK_PID"
echo "ngrok Web Interface: http://127.0.0.1:4040"

# Wait indefinitely until a termination signal is received
wait
