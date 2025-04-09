import asyncio
import json
import websockets
import serial
import time

# Set up the serial connection to Arduino.
print("Setting up serial connection")
ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
print("ser = ", ser)
ser.reset_input_buffer()
print("ser after reset = ", ser)

def read_base_angle_from_arduino():
    """
    Attempt to read the initial base angle from the Arduino.
    Expecting a JSON string such as {"pitch":0,"yaw":0}.
    """
    try:
        print("Before readline() -", ser.readline())
        line = ser.readline().decode('utf-8').rstrip()
        print(f"Read line: {line}")
        if line:
            data = json.loads(line)
            return data
    except Exception as e:
        print("Error reading base angle:", e)
    print("Defaulting base angle to 0, 0")
    return {"pitch": 0, "yaw": 0}

# Read the base angle once at startup.
BASE_ANGLE = read_base_angle_from_arduino()
print("Read base angle from Arduino:", BASE_ANGLE)

async def handler(websocket):
    # Send the base angle once when a client connects.
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

            # Process only messages with both "pitch" and "yaw" keys.
            if isinstance(received_data, dict) and "pitch" in received_data and "yaw" in received_data:
                print("Received updated angle from client:", received_data)
                
                # Send waiting status to the client.
                waiting_packet = {"status": "waiting"}
                await websocket.send(json.dumps(waiting_packet))
                print("Sent waiting status:", waiting_packet)
                
                # Send the updated angle command to Arduino.
                command = json.dumps(received_data) + "\n"
                ser.write(command.encode('utf-8'))
                print("Sent to Arduino:", command.strip())
                
                # Wait for Arduino to process the command and send back encoder values.
                # Wait for non-empty response for up to 10 seconds.
                arduino_response = b""
                start_time = time.time()
                while not arduino_response and (time.time() - start_time) < 10:
                    arduino_response = await asyncio.to_thread(ser.readline)
                
                if not arduino_response:
                    print("No response from Arduino within timeout.")
                    encoder_data = {"pitch": received_data["pitch"], "yaw": received_data["yaw"]}
                else:
                    print("ARDUINO RESPONSE BEFORE DECODE:", arduino_response)
                    arduino_response_decoded = arduino_response.decode('utf-8').rstrip()
                    print("Arduino response:", arduino_response_decoded)
                    try:
                        encoder_data = json.loads(arduino_response_decoded)
                    except Exception as e:
                        print("Error parsing Arduino response:", e)
                        encoder_data = {"pitch": received_data["pitch"], "yaw": received_data["yaw"]}
                
                # Send a ready packet to the client with the encoder values.
                ready_packet = {
                    "status": "ready",
                    "pitch": encoder_data.get("pitch", received_data["pitch"]),
                    "yaw": encoder_data.get("yaw", received_data["yaw"])
                }
                await websocket.send(json.dumps(ready_packet))
                print("Sent ready packet:", ready_packet)
            else:
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
