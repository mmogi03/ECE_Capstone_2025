#!/usr/bin/env bash
# run_services.sh
# Launches main.py and starts localtunnel with a fixed subdomain.

set -e

PORT=8765
SUBDOMAIN="car-rutgers"

# 1. Kill any process on the port
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null; then
    echo "Port ${PORT} is in use. Killing process(es)..."
    kill -9 $(lsof -t -i:${PORT})
    sleep 1
fi

# 2. Trap SIGINT/SIGTERM to clean up child processes
trap "echo 'Stopping services...'; kill $MAIN_PID $LT_PID; exit 0" SIGINT SIGTERM

# 3. Start main.py
echo "Starting main Python process (main.py)..."
python3 main.py &
MAIN_PID=$!

# Give main.py a moment
sleep 2

# 4. Start localtunnel with just the subdomain
echo "Starting localtunnel on port ${PORT} using subdomain ${SUBDOMAIN}..."
lt --port ${PORT} --subdomain ${SUBDOMAIN} &
LT_PID=$!

echo ""
echo "🚀  Tunnel should be available at: https://${SUBDOMAIN}.loca.lt"
echo "Main process PID: $MAIN_PID"
echo "localtunnel PID:  $LT_PID"
echo ""
echo "Press Ctrl+C to stop services."

# 5. Wait forever
wait
