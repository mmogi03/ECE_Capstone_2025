#!/bin/bash

# Check if port 8765 is in use and kill any process using it
if lsof -Pi :8765 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 8765 is in use. Killing process(es)..."
    kill -9 $(lsof -t -i:8765)
    sleep 1
fi

# Trap Ctrl+C (SIGINT) and SIGTERM to gracefully kill child processes
trap "echo 'Stopping services...'; kill $PYTHON_PID $NGROK_PID; exit 0" SIGINT SIGTERM

echo "Starting Python WebSocket server..."
# Start the Python WebSocket server in the background
python3 websocket_server_with_connection_to_arduino.py &
PYTHON_PID=$!

# Give the Python server a moment to initialize
sleep 2

echo "Starting ngrok tunnel on port 8765..."
# Start ngrok in the background with the proper host-header flag
ngrok http 8765 --host-header=localhost:8765 &
NGROK_PID=$!

echo "Services started:"
echo " - Python WebSocket server PID: $PYTHON_PID"
echo " - ngrok PID: $NGROK_PID"
echo "ngrok Web Interface: http://127.0.0.1:4040"

# Wait indefinitely until a termination signal is received
wait
