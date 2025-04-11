import asyncio
import json
import websockets
import time
from multiprocessing import Process, Value
from ctypes import c_bool

from serial_interface import SerialInterface
from dl_inference import run_inference

# =============================================================================
# Persistent Serial Connection Setup for Base Angle Reading and Commands
# =============================================================================
# Create a persistent SerialInterface that remains open throughout the server’s lifetime.
persistent_serial_intf = SerialInterface()
print("Persistent serial connection established.")

def read_base_angle_from_arduino(serial_intf):
    """
    Attempt to read the initial base angle from the Arduino using SerialInterface.
    Expecting a JSON string such as {"pitch":0,"yaw":0}.
    """
    try:
        print("Attempting to read base angle from Arduino")
        line = serial_intf.read_response()
        print(f"Read line: {line}")
        if line:
            return line
    except Exception as e:
        print("Error reading base angle:", e)
    print("Defaulting base angle to 0, 0")
    return {"pitch": 0, "yaw": 0}

# Read the base angle once at startup using the persistent connection.
BASE_ANGLE = read_base_angle_from_arduino(persistent_serial_intf)
print("Read base angle from Arduino:", BASE_ANGLE)

# =============================================================================
# Shared DL Inference Control Variables
# =============================================================================
auto_adjust_flag = Value(c_bool, True)
dl_process = None  # Global handle for the DL inference process.

# =============================================================================
# WebSocket Handler
# =============================================================================
async def handler(websocket):
    global auto_adjust_flag, dl_process, persistent_serial_intf
    # When a client connects, send the base angle.
    await websocket.send(json.dumps(BASE_ANGLE))
    print("Sent base angle:", BASE_ANGLE)
    
    while True:
        try:
            message = await websocket.recv()
            data = json.loads(message)
            print("Received message from client:", data)
            
            # -----------------------------------------------------------------
            # Handle autoAdjust commands.
            # -----------------------------------------------------------------
            if "autoAdjust" in data:
                if data["autoAdjust"] == "False":
                    auto_adjust_flag.value = False
                    # Terminate the DL inference process if active.
                    if dl_process is not None and dl_process.is_alive():
                        dl_process.terminate()
                        dl_process.join(timeout=5)
                        if dl_process.is_alive():
                            dl_process.kill()
                        dl_process = None
                    print("Auto adjust disabled by client.")
                elif data["autoAdjust"] == "True":
                    auto_adjust_flag.value = True
                    # Start the DL inference process if it is not running.
                    if dl_process is None or not dl_process.is_alive():
                        dl_process = Process(target=run_inference, args=(auto_adjust_flag,))
                        dl_process.start()
                    print("Auto adjust enabled by client.")
            
            # -----------------------------------------------------------------
            # Handle explicit angle updates (only when autoAdjust is off).
            # The packet should contain "pitch" and "yaw".
            # -----------------------------------------------------------------
            if isinstance(data, dict) and "pitch" in data and "yaw" in data and not auto_adjust_flag.value:
                print("Received updated angle from client:", data)
                
                # Send waiting status to the client.
                waiting_packet = {"status": "waiting"}
                await websocket.send(json.dumps(waiting_packet))
                print("Sent waiting status:", waiting_packet)
                
                # Use the persistent SerialInterface to send the updated angle command.
                command = json.dumps({"pitch": data["pitch"], "yaw": data["yaw"]})
                persistent_serial_intf.send_command(command)
                print("Sent to Arduino:", command)
                
                # Wait for Arduino response for up to 10 seconds.
                arduino_response = None
                start_time = time.time()
                while (time.time() - start_time) < 10:
                    arduino_response = await asyncio.to_thread(persistent_serial_intf.read_response)
                    if arduino_response:
                        break
                    await asyncio.sleep(0.1)
                
                if not arduino_response:
                    print("No response from Arduino within timeout.")
                    encoder_data = {"pitch": data["pitch"], "yaw": data["yaw"]}
                else:
                    print("Arduino response:", arduino_response)
                    encoder_data = (
                        arduino_response
                        if isinstance(arduino_response, dict)
                        else {"pitch": data["pitch"], "yaw": data["yaw"]}
                    )
                
                # Send a ready packet with the encoder values.
                ready_packet = {
                    "status": "ready",
                    "pitch": encoder_data.get("pitch", data["pitch"]),
                    "yaw": encoder_data.get("yaw", data["yaw"])
                }
                await websocket.send(json.dumps(ready_packet))
                print("Sent ready packet:", ready_packet)
            else:
                # For any non-angle update messages.
                print("Received non-angle update message:", data)
            
        except websockets.ConnectionClosed:
            print("Client disconnected")
            break
        except Exception as e:
            print("Error in handler:", e)

# =============================================================================
# Main Event Loop
# =============================================================================
async def main():
    global auto_adjust_flag, dl_process
    # If auto_adjust_flag is initially True, start the DL inference process.
    if auto_adjust_flag.value:
        dl_process = Process(target=run_inference, args=(auto_adjust_flag,))
        dl_process.start()
    
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server started on port 8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
