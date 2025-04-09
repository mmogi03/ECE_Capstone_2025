import asyncio
import json
import websockets

# Define the base angle that will be sent to the client upon connection.
BASE_ANGLE = {"pitch": 0, "yaw": 0}

async def handler(websocket):
    # Send the base angle once when a client connects.
    await websocket.send(json.dumps(BASE_ANGLE))
    print("Sent base angle:", BASE_ANGLE)

    while True:
        try:
            # Wait for the client to send an updated angle.
            message = await websocket.recv()
            print("Received message:", message)

            try:
                received_data = json.loads(message)
            except json.JSONDecodeError:
                print("Invalid JSON received:", message)
                continue

            # Process the update only if both "pitch" and "yaw" keys are present.
            if isinstance(received_data, dict) and "pitch" in received_data and "yaw" in received_data:
                # Print the updated angle received
                print("Received updated angle from client:", received_data)
                
                # Send waiting status to the client.
                waiting_packet = {"status": "waiting"}
                await websocket.send(json.dumps(waiting_packet))
                print("Sent waiting status:", waiting_packet)
                
                # Simulate delay representing motor update time.
                await asyncio.sleep(8)
                
                # After delay, send ready status without the angle (client already has it).
                ready_packet = {"status": "ready"}
                await websocket.send(json.dumps(ready_packet))
                print("Sent ready packet:", ready_packet)
            else:
                # For other messages, log them.
                print("Received non-angle update message:", received_data)
        
        except websockets.ConnectionClosed:
            print("Client disconnected")
            break

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server started on port 8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
