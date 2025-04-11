import numpy as np
import cv2 as cv
import dlib
import math
import json
from time import sleep, time
from picamera2 import Picamera2
from sympy import Plane, Point3D
from utils import DLT, get_projection_matrix  # your existing utility functions
from serial_interface import SerialInterface

def run_inference(auto_adjust_flag):
    """
    This function runs in a separate process. While auto_adjust_flag.value is True,
    it continuously captures stereo images, computes the 3D position using the 28th facial landmark,
    derives pitch and yaw using a difference-vector method, sends these angles to Arduino over serial,
    and then waits until it receives a response from Arduino before proceeding.
    """
    serial_intf = SerialInterface()  # create a serial connection
    frame_shape = [1232, 1640]

    # Initialize Picamera2 for both cameras.
    picam2_left = Picamera2(camera_num=0)
    config_left = picam2_left.create_preview_configuration(
        main={"size": (frame_shape[1], frame_shape[0])}
    )
    picam2_left.configure(config_left)
    picam2_left.start()

    picam2_right = Picamera2(camera_num=1)
    config_right = picam2_right.create_preview_configuration(
        main={"size": (frame_shape[1], frame_shape[0])}
    )
    picam2_right.configure(config_right)
    picam2_right.start()

    # Retrieve projection matrices from calibration.
    P_left = get_projection_matrix(0)
    P_right = get_projection_matrix(1)

    # Hard–set 3D rear window point (in homogeneous coordinates), then convert to a 3D tuple.
    threeD_window_pt = [
        [-5e+05],
        [8.65304694e+04],
        [-1.1e+04],
        [1.0]
    ]
    window_pt = (threeD_window_pt[0][0], threeD_window_pt[1][0], threeD_window_pt[2][0])

    # Initialize dlib.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while auto_adjust_flag.value:
        # Capture frames from each camera.
        frame_left = picam2_left.capture_array()
        frame_right = picam2_right.capture_array()

        if frame_left is None or frame_right is None:
            sleep(0.1)
            continue

        gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

        faces_left = detector(gray_left)
        faces_right = detector(gray_right)

        if len(faces_left) > 0 and len(faces_right) > 0:
            shape_left = predictor(gray_left, faces_left[0])
            shape_right = predictor(gray_right, faces_right[0])
            midpt_left = (shape_left.part(27).x, shape_left.part(27).y)
            midpt_right = (shape_right.part(27).x, shape_right.part(27).y)
            threeD_midpt = DLT(P_left, P_right, midpt_left, midpt_right)
            if len(threeD_midpt) == 4:
                threeD_midpt = [threeD_midpt[i] / threeD_midpt[3] for i in range(3)]
            
            # Compute a difference vector (assuming Z is forward).
            diff_vec = np.array(window_pt) - np.array(threeD_midpt)
            yaw_infer = math.degrees(math.atan2(diff_vec[0], diff_vec[2]))
            pitch_infer = math.degrees(math.atan2(diff_vec[1], diff_vec[2]))
            
            # Prepare and send the command to Arduino.
            command = json.dumps({"pitch": pitch_infer, "yaw": yaw_infer})
            serial_intf.send_command(command)
            print("DL inference sent angles:", command)
            
            # Wait for Arduino response before proceeding.
            response = None
            timeout = 12  # seconds (adjust based on Arduino delay; your code uses delay(8000) = 8 sec)
            start_time = time()
            while response is None and (time() - start_time) < timeout:
                response = serial_intf.read_response()
                if response is not None:
                    break
                sleep(0.1)
            if response is not None:
                print("Received response from Arduino:", response)
            else:
                print("No response received from Arduino within timeout.")
        else:
            print("DL inference: Face not detected.")

        sleep(0.1)  # adjust loop speed as desired

    # Cleanup
    picam2_left.stop()
    picam2_right.stop()
    serial_intf.close()
