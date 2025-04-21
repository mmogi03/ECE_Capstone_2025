# serial_interface.py
import serial
import json
import time
from multiprocessing import Lock

# Create a global lock for serial access.
global_serial_lock = Lock()

class SerialInterface:
    def __init__(self, port="/dev/ttyACM0", baudrate=9600, timeout=0.5, lock=global_serial_lock):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.lock = lock
        # Allow time for the connection to initialize.
        time.sleep(2)
        
    def send_command(self, command):
        """Send a command string to Arduino with mutual exclusion."""
        if not command.endswith("\n"):
            command += "\n"
        with self.lock:
            self.ser.write(command.encode('utf-8'))
        
    def read_response(self):
        """Read a response line from Arduino under mutual exclusion and attempt to parse JSON."""
        with self.lock:
            line = self.ser.readline().decode('utf-8').rstrip()
        if line:
            try:
                return json.loads(line)
            except Exception:
                return line
        return None

    def close(self):
        with self.lock:
            self.ser.close()
