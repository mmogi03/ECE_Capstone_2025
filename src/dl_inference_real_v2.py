#!/usr/bin/env python3
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

log_ind = 0
read_arduino_count = 0

"""
Hard-coded measurements required in the angle computations.
Assume this is known based on the car dimensions.
"""
# d1 = 58.42  # Example offset in X
# d2 = 38.1   # Example offset in Y
# dwz = 330.2
# dwy = 35.56

d1 = 51.5
d2 = 32.0
dwz = 178.0
dwy = 17.0

# Display scale factor for visualization (0.5 = half-size windows)
DISPLAY_SCALE = 0.5

def run_inference(auto_adjust_flag):
    """
    This function runs in a separate process. While auto_adjust_flag.value is True,
    it continuously captures stereo images, computes the 3D position using the 28th facial landmark,
    derives pitch and yaw using a difference-vector method, sends these angles to Arduino over serial,
    and then waits until it receives a response from Arduino before proceeding.
    """
    global log_ind
    global read_arduino_count
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
        [d1],
        [dwy],
        [dwz],
        [1.00000000e+00]
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
                print(f"Rearview mirror midpoint 3D {threeD_midpt}")


            # ── Visualization ──
            vis_left = cv.cvtColor(frame_left, cv.COLOR_BGR2RGB)
            vis_right = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB)
            for i in range(shape_left.num_parts):
                pL = (shape_left.part(i).x, shape_left.part(i).y)
                pR = (shape_right.part(i).x, shape_right.part(i).y)
                if i == 27:
                    cv.drawMarker(vis_left, pL, (255,0,0),
                                  markerType=cv.MARKER_TRIANGLE_UP,
                                  markerSize=10, thickness=2)
                    cv.drawMarker(vis_right, pR, (255,0,0),
                                  markerType=cv.MARKER_TRIANGLE_UP,
                                  markerSize=10, thickness=2)
                else:
                    cv.circle(vis_left, pL, radius=2, color=(0,255,0), thickness=-1)
                    cv.circle(vis_right, pR, radius=2, color=(0,255,0), thickness=-1)
            # down-sample for display
            h, w = vis_left.shape[:2]
            disp_size = (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE))
            cv.imshow('Left Camera',  cv.resize(vis_left,  disp_size))
            cv.imshow('Right Camera', cv.resize(vis_right, disp_size))
            cv.waitKey(1)
            
            # Compute a difference vector (assuming Z is forward).
            face_pt_mirror = (
                threeD_midpt[0] + d1,
                threeD_midpt[1] + d2,
                threeD_midpt[2]
            )
            window_pt_mirror = (
                window_pt[0] - d1,
                window_pt[1],
                window_pt[2]
            )

            driver_pt_xz = np.array([face_pt_mirror[0], 0, face_pt_mirror[2]])
            rear_window_pt_xz = np.array([window_pt_mirror[0], 0, window_pt_mirror[2]])
            xy_plane_pt_xz = np.array([1, 0, 0])
            alpha = math.acos(np.dot(driver_pt_xz, rear_window_pt_xz) / (np.linalg.norm(driver_pt_xz) * np.linalg.norm(rear_window_pt_xz)))
            beta = math.acos(np.dot(driver_pt_xz, xy_plane_pt_xz) / np.linalg.norm(driver_pt_xz) * (np.linalg.norm(xy_plane_pt_xz)))
            yaw_infer = (90 - math.degrees(alpha / 2 + beta))*17.9

            driver_pt_yz = np.array([0, face_pt_mirror[1], face_pt_mirror[2]])
            rear_window_pt_yz = np.array([0, window_pt_mirror[1], window_pt_mirror[2]])
            xz_plane_pt_yz = np.array([0, 0, 1])

            theta = math.acos(np.dot(xz_plane_pt_yz, rear_window_pt_yz) / (np.linalg.norm(xz_plane_pt_yz) * np.linalg.norm(rear_window_pt_yz)))
            rho = math.acos(np.dot(rear_window_pt_yz, driver_pt_yz) / (np.linalg.norm(rear_window_pt_yz) * np.linalg.norm(driver_pt_yz)))
            pitch_infer = math.degrees(rho / 2 + theta)*17.9*-1

            # Prepare and send the command to Arduino.
            command = json.dumps({"pitch": pitch_infer, "yaw": yaw_infer})
            serial_intf.send_command(command)
            print(f"[{log_ind}] DL inference sent angles:", command)
            log_ind += 1
            
            # Wait for Arduino response before proceeding.
            # print("SLEEPING FOR 45s")
            # sleep(45)
            print(f"[{log_ind}] Waiting for Arduino response...")
            log_ind += 1

            response = None
            timeout = 12  # seconds (adjust based on Arduino delay; your code uses delay(8000) = 8 sec)
            start_time = time()
            while response is None:      
                print(f"[{log_ind}] !!waiting on response")      
                log_ind += 1
                response = serial_intf.read_response() if read_arduino_count == 0 else serial_intf.read_response()
                read_arduino_count += 1

                if response is not None:
                    break
                sleep(0.1)
            if response is not None:
                print(f"[{log_ind}] Received response from Arduino:", response)
                log_ind += 1
                sleep(40)
            else:
                print(f"[{log_ind}] No response received from Arduino within timeout.")
                log_ind += 1
        else:
            # print(f"[{log_ind}] DL inference: Face not detected.")
            log_ind += 1

        sleep(0.1)  # adjust loop speed as desired

    # Cleanup
    picam2_left.stop()
    picam2_right.stop()
    serial_intf.close()
