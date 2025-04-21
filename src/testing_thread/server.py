import asyncio
import json
import websockets
import time
from multiprocessing import Process, Value
from ctypes import c_bool

from serial_interface import SerialInterface
from dl_inference import run_inference

# Shared flag to control DL inference.
auto_adjust_flag = Value(c_bool, True)
dl_process = None  # global handle for the inference process

async def handler(websocket):
    global auto_adjust_flag, dl_process
    # Send a welcome message.
    await websocket.send(json.dumps({"status": "connected"}))
    
    while True:
        try:
            message = await websocket.recv()
            data = json.loads(message)
            print("Received message from client:", data)
            
            # Check for auto adjust commands.
            if "autoAdjust" in data:
                if data["autoAdjust"] == "False":
                    auto_adjust_flag.value = False
                    # If the DL inference process is running, terminate it.
                    if dl_process is not None and dl_process.is_alive():
                        dl_process.terminate()
                        dl_process.join()
                        dl_process = None
                    print("Auto adjust disabled by client.")
                elif data["autoAdjust"] == "True":
                    auto_adjust_flag.value = True
                    # Start the DL inference process if not running.
                    if dl_process is None or not dl_process.is_alive():
                        dl_process = Process(target=run_inference, args=(auto_adjust_flag,))
                        dl_process.start()
                    print("Auto adjust enabled by client.")
            
            # If the client sends explicit angle updates (when autoAdjust is off).
            if "pitch" in data and "yaw" in data and not auto_adjust_flag.value:
                serial_intf = SerialInterface()  # open a temporary serial connection
                command = json.dumps({"pitch": data["pitch"], "yaw": data["yaw"]})
                serial_intf.send_command(command)
                print("Sent client-updated angles to Arduino:", command)
                # Read response from Arduino:
                response = serial_intf.read_response()
                serial_intf.close()
                await websocket.send(json.dumps({"status": "command_sent", "response": response}))
            
        except websockets.ConnectionClosed:
            print("Client disconnected")
            break

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
