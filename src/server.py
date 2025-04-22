import asyncio
import json
import logging
import time
import websockets
from multiprocessing import Process, Value
from ctypes import c_bool

from serial_interface import SerialInterface
from dl_inference_real_v2 import run_inference

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Persistent Serial Connection Setup for Base Angle Reading and Commands
# =============================================================================
persistent_serial_intf = SerialInterface()
logger.info("Persistent serial connection established.")

def read_base_angle_from_arduino(serial_intf):
    """
    Attempt to read the initial base angle from the Arduino using SerialInterface.
    Expecting a JSON string such as {"pitch":0,"yaw":0}.
    Strips any stray carriage-returns so each log prints on its own line.
    Returns a dict.
    """
    try:
        logger.info("Attempting to read base angle from Arduino")
        raw = serial_intf.read_response()
        if raw:
            if isinstance(raw, str):
                clean = raw.replace('\r', '').replace('\n', '').strip()
                logger.info(f"Read line: {clean}")
                return json.loads(clean)
            elif isinstance(raw, dict):
                logger.info(f"Read dict: {raw}")
                return raw
    except Exception as e:
        logger.error("Error reading base angle: %s", e)
    # Fallback if nothing read
    logger.warning("Defaulting base angle to 0,0")
    return {"pitch": 0, "yaw": 0}

# Read base angle once at startup
BASE_ANGLE = read_base_angle_from_arduino(persistent_serial_intf)
logger.info("Base angle from Arduino: %s", BASE_ANGLE)

# =============================================================================
# Shared DL Inference Control Variables
# =============================================================================
auto_adjust_flag = Value(c_bool, False)
dl_process = None  # Global handle for the DL inference process.

# =============================================================================
# WebSocket Handler
# =============================================================================
async def handler(websocket):
    global auto_adjust_flag, dl_process, persistent_serial_intf

    # Send the base angle on new connection
    await websocket.send(json.dumps(BASE_ANGLE))
    logger.info("Sent base angle: %s", BASE_ANGLE)

    while True:
        try:
            message = await websocket.recv()
            data = json.loads(message)
            logger.info("Received message from client: %r", data)

            # -----------------------------------------------------------------
            # Handle autoAdjust commands.
            # -----------------------------------------------------------------
            if "autoAdjust" in data:
                if data["autoAdjust"] == "False":
                    logger.info("AUTO ADJUST IS FALSE")
                    auto_adjust_flag.value = False
                    if dl_process and dl_process.is_alive():
                        dl_process.terminate()
                        dl_process.join(timeout=5)
                        if dl_process.is_alive():
                            dl_process.kill()
                        dl_process = None
                    logger.info("Auto adjust disabled by client.")
                elif data["autoAdjust"] == "True":
                    logger.info("AUTO ADJUST IS TRUE")
                    auto_adjust_flag.value = True
                    if not dl_process or not dl_process.is_alive():
                        dl_process = Process(target=run_inference, args=(auto_adjust_flag,))
                        dl_process.start()
                    logger.info("Auto adjust enabled by client.")

            # -----------------------------------------------------------------
            # Handle explicit angle updates (only when autoAdjust is off).
            # -----------------------------------------------------------------
            if (
                isinstance(data, dict)
                and "pitch" in data
                and "yaw" in data
                and not auto_adjust_flag.value
            ):
                logger.info("Received updated angle from client: %s", data)

                # Notify client we're waiting for Arduino
                waiting_packet = {"status": "waiting"}
                await websocket.send(json.dumps(waiting_packet))
                logger.info("Sent waiting status: %s", waiting_packet)

                # Send command to Arduino
                cmd = json.dumps({"pitch": data["pitch"] * 17.9 * -1, "yaw": data["yaw"] * 17.9})
                persistent_serial_intf.send_command(cmd)
                logger.info("Sent to Arduino: %s", cmd)

                # Await response, up to 10s
                raw = None
                start_time = time.time()
                while time.time() - start_time < 10:
                    raw = await asyncio.to_thread(persistent_serial_intf.read_response)
                    if raw:
                        break
                    await asyncio.sleep(0.1)

                # Process Arduino response
                if not raw:
                    logger.warning("No response from Arduino within timeout.")
                    encoder_data = {"pitch": data["pitch"], "yaw": data["yaw"]}
                else:
                    if isinstance(raw, str):
                        clean = raw.replace('\r', '').replace('\n', '').strip()
                        logger.info("Arduino response (str): %s", clean)
                        try:
                            encoder_data = json.loads(clean)
                        except json.JSONDecodeError:
                            logger.error("JSON decode error on Arduino response, using sent values")
                            encoder_data = {"pitch": data["pitch"], "yaw": data["yaw"]}
                    elif isinstance(raw, dict):
                        logger.info("Arduino response (dict): %s", raw)
                        encoder_data = raw
                    else:
                        logger.warning("Unexpected Arduino response type %s, using sent values", type(raw))
                        encoder_data = {"pitch": data["pitch"], "yaw": data["yaw"]}

                # Send ready packet
                ready_packet = {
                    "status": "ready",
                    "pitch": encoder_data.get("pitch", data["pitch"]),
                    "yaw": encoder_data.get("yaw", data["yaw"])
                }
                await websocket.send(json.dumps(ready_packet))
                logger.info("Sent ready packet: %s", ready_packet)

            else:
                logger.debug("Non-angle-update message or autoAdjust is on.")

        except websockets.ConnectionClosed:
            logger.info("Client disconnected")
            break
        except Exception:
            logger.exception("Error in handler:")

# =============================================================================
# Main Event Loop
# =============================================================================
async def main():
    global auto_adjust_flag, dl_process
    if auto_adjust_flag.value:
        dl_process = Process(target=run_inference, args=(auto_adjust_flag,))
        dl_process.start()
    async with websockets.serve(handler, "0.0.0.0", 8765):
        logger.info("WebSocket server started on port 8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
