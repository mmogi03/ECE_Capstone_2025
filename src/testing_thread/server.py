import asyncio
import json
import websockets

async def handler(websocket, auto_adjust_flag):
    # Define the base angle and send it once.
    BASE_ANGLE = {"pitch": 0, "yaw": 0}
    await websocket.send(json.dumps(BASE_ANGLE))
    print("Sent base angle:", BASE_ANGLE)

    while True:
        try:
            message = await websocket.recv()
            print("Received message:", message)
            try:
                received_data = json.loads(message)
            except json.JSONDecodeError:
                print("Invalid JSON received:", message)
                continue

            # Process only messages with both "pitch" and "yaw".
            if isinstance(received_data, dict) and "pitch" in received_data and "yaw" in received_data:
                print("Received updated angle from client:", received_data)
                # Update the shared auto_adjust_flag from the payload.
                auto_adjust_flag.value = received_data.get("autoAdjust", False)
                
                # Send waiting status.
                waiting_packet = {"status": "waiting"}
                await websocket.send(json.dumps(waiting_packet))
                print("Sent waiting status:", waiting_packet)
                
                # Simulate motor update delay.
                await asyncio.sleep(8)
                
                # After delay, send ready status (no angle data; client already has it).
                ready_packet = {"status": "ready"}
                await websocket.send(json.dumps(ready_packet))
                print("Sent ready packet:", ready_packet)
            else:
                print("Received non-angle update message:", received_data)
        except websockets.ConnectionClosed:
            print("Client disconnected")
            break

async def server_main(auto_adjust_flag):
    async with websockets.serve(lambda ws, path: handler(ws, auto_adjust_flag), "0.0.0.0", 8765):
        print("WebSocket server started on port 8765")
        await asyncio.Future()  # run forever

def run_server(auto_adjust_flag):
    asyncio.run(server_main(auto_adjust_flag))
