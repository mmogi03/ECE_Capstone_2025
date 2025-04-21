#!/usr/bin/env python3
"""
Stereo‑based rear‑view‑mirror auto‑adjust: full inference script
----------------------------------------------------------------
 • Captures synchronized frames from two Raspberry Pi cameras
 • Finds the 28‑th facial landmark (nose bridge midpoint) in each image
 • Triangulates its 3‑D position with Direct Linear Transform (DLT)
 • Converts that point into the mirror’s local coordinate frame
 • Computes yaw and pitch with the plane‑bisector / law‑of‑reflection method
 • Sends the angles as JSON over a serial link to an Arduino
 • Waits for the Arduino to finish the motor movement before continuing

Author:  ⓒ 2025 Your Team
"""

import math
import json
from time import sleep, time

import cv2 as cv
import dlib
import numpy as np
from picamera2 import Picamera2
from sympy import Plane, Point3D

from utils import DLT, get_projection_matrix          # your existing utilities
from serial_interface import SerialInterface          # your existing serial helper

# ────────────────────────────────────────────────────────────────────────────────
# Tunable parameters (edit to suit your installation)
# ────────────────────────────────────────────────────────────────────────────────
FRAME_SHAPE        = (1232, 1640)   # (height, width) of camera output
D1_OFFSET          = 70.0           # X‑offset from camera origin → mirror origin
D2_OFFSET          = 71.0           # Y‑offset from camera origin → mirror origin
REAR_WINDOW_Z      = 141.0          # Z coordinate of the rear window in mirror frame
ARDUINO_TIMEOUT    = 12             # seconds to wait for a response
LOOP_SLEEP         = 0.10           # base loop delay (s) when all goes well

# Derived constant: fixed 3‑D point on the rear window in *camera* frame
REAR_WINDOW_PT_CAM = (D1_OFFSET, D2_OFFSET, REAR_WINDOW_Z, 1.0)

# ────────────────────────────────────────────────────────────────────────────────
# Helper: compute mirror yaw & pitch by plane geometry
# ────────────────────────────────────────────────────────────────────────────────
def compute_pitch_yaw(face_pt_cam: tuple) -> tuple[float, float]:
    """
    face_pt_cam : (x, y, z) of the driver’s facial landmark in the *camera* frame
    returns     : (pitch_deg, yaw_deg) for the mirror
    """
    # --- transform the points into the mirror’s coordinate frame ----------------
    face_pt_mirror = (
        face_pt_cam[0] + D1_OFFSET,
        face_pt_cam[1] + D2_OFFSET,
        face_pt_cam[2],
    )
    window_pt_mirror = (
        REAR_WINDOW_PT_CAM[0] - D1_OFFSET,
        REAR_WINDOW_PT_CAM[1] - D2_OFFSET - 1e‑3,   # epsilon keeps planes distinct
        REAR_WINDOW_PT_CAM[2],
    )

    # --- build planes -----------------------------------------------------------
    driver_plane = Plane(
        Point3D(0, 0, 0),
        Point3D(face_pt_mirror[0], 0, face_pt_mirror[2]),
        Point3D(face_pt_mirror),
    )
    rear_window_plane = Plane(
        Point3D(0, 0, 0),
        Point3D(window_pt_mirror[0], 0, window_pt_mirror[2]),
        Point3D(window_pt_mirror),
    )
    xy_plane = Plane(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
    xz_plane = Plane(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))

    # --- yaw: bisector between driver‑ and rear‑window planes -------------------
    alpha = driver_plane.angle_between(rear_window_plane)
    beta  = driver_plane.angle_between(xy_plane)
    yaw   = math.degrees(alpha / 2 + beta)

    # --- pitch: angle between mirror & xz plane ---------------------------------
    driver_mirror_plane = Plane(Point3D(0, 0, 0), Point3D(face_pt_mirror),
                                Point3D(window_pt_mirror))
    pitch = math.degrees(driver_mirror_plane.angle_between(xz_plane))
    return pitch, yaw


# ────────────────────────────────────────────────────────────────────────────────
# Inference loop – designed to be started in its own process
# ────────────────────────────────────────────────────────────────────────────────
def run_inference(auto_adjust_flag):
    """
    Continuously adjust the mirror while *auto_adjust_flag.value* is True.
    Intended to be launched in a separate multiprocessing.Process.
    """
    log_id = 0
    arduino_reads = 0

    # Serial link to Arduino
    ser = SerialInterface()

    # ─ Camera setup ─────────────────────────────────────────────────────────────
    cam_left  = Picamera2(camera_num=0)
    cam_right = Picamera2(camera_num=1)

    cfg_left  = cam_left .create_preview_configuration(main={"size": (FRAME_SHAPE[1],
                                                                     FRAME_SHAPE[0])})
    cfg_right = cam_right.create_preview_configuration(main={"size": (FRAME_SHAPE[1],
                                                                     FRAME_SHAPE[0])})
    cam_left .configure(cfg_left);  cam_left .start()
    cam_right.configure(cfg_right); cam_right.start()

    # ─ Load projection matrices ────────────────────────────────────────────────
    P_left  = get_projection_matrix(0)
    P_right = get_projection_matrix(1)

    # ─ dlib face detector & predictor ──────────────────────────────────────────
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # ─ Main loop ───────────────────────────────────────────────────────────────
    while auto_adjust_flag.value:
        frame_L = cam_left .capture_array()
        frame_R = cam_right.capture_array()
        if frame_L is None or frame_R is None:
            sleep(LOOP_SLEEP)
            continue

        gray_L = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)
        gray_R = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)

        faces_L = detector(gray_L)
        faces_R = detector(gray_R)

        if faces_L and faces_R:
            # first face in each image
            shape_L = predictor(gray_L, faces_L[0])
            shape_R = predictor(gray_R, faces_R[0])

            # 28‑th landmark (index 27)
            mid_L = (shape_L.part(27).x, shape_L.part(27).y)
            mid_R = (shape_R.part(27).x, shape_R.part(27).y)

            # Triangulate
            pt3d_h = DLT(P_left, P_right, mid_L, mid_R)
            if len(pt3d_h) != 4 or pt3d_h[3] == 0:
                print(f"[{log_id}] Degenerate 3‑D point from DLT.")
                log_id += 1
                sleep(LOOP_SLEEP)
                continue
            pt3d_cam = [pt3d_h[i] / pt3d_h[3] for i in range(3)]
            print(f"3‑D landmark (camera frame): {pt3d_cam}")

            try:
                pitch, yaw = compute_pitch_yaw(pt3d_cam)
                cmd = json.dumps({"pitch": pitch, "yaw": yaw})
            except Exception as exc:
                print(f"[{log_id}] Geometry error: {exc}")
                log_id += 1
                sleep(LOOP_SLEEP)
                continue

            # ─ Send command and wait for acknowledgment ────────────────────────
            ser.send_command(cmd)
            print(f"[{log_id}] → Arduino: {cmd}")
            log_id += 1

            print(f"[{log_id}] Waiting for Arduino…")
            log_id += 1
            start = time()
            reply = None
            while (time() - start) < ARDUINO_TIMEOUT:
                reply = ser.read_response()
                arduino_reads += 1
                if reply:
                    break
                sleep(0.05)

            if reply:
                print(f"[{log_id}] ← Arduino: {reply}")
            else:
                print(f"[{log_id}] No response within {ARDUINO_TIMEOUT}s.")
            log_id += 1

        else:
            print(f"[{log_id}] Face not detected in both cameras.")
            log_id += 1

        sleep(LOOP_SLEEP)

    # ─ Cleanup ─────────────────────────────────────────────────────────────────
    cam_left.stop();  cam_right.stop();  ser.close()
    print("Inference loop terminated cleanly.")

# ────────────────────────────────────────────────────────────────────────────────
# If you need a stand‑alone script, remove *auto_adjust_flag* and call run_inference()
# directly. When using multiprocessing, spawn this function with a shared flag.
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from multiprocessing import Value
    flag = Value('b', True)          # simple shared bool
    try:
        run_inference(flag)
    except KeyboardInterrupt:
        flag.value = False
