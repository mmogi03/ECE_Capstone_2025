import numpy as np
import cv2 as cv
import dlib
import math
import serial
from time import sleep
from picamera2 import Picamera2
from sympy import Plane, Point3D
from utils import DLT, get_projection_matrix, read_rotation_translation

def main():
    # >>> DEFINE YOUR OFFSETS HERE <<<
    # Distance in X and Y (in whatever units you are working with) 
    # from the camera to the mirror’s coordinate origin.
    alpha_d2 = 20.0 # threshold for d2
    d1 = 100.0  # Example offset in X
    d2 = 50.0  - alpha_d2 # Example offset in Y

    # Define desired frame resolution (height, width)
    frame_shape = [1232, 1640]

    # Initialize Picamera2 objects for the stereo cameras.
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

    # Retrieve projection matrices from your saved calibration.
    P_left = get_projection_matrix(0)
    P_right = get_projection_matrix(1)

    # (Optionally, you can read extrinsics from camera0 for reference.)
    R0, t0 = read_rotation_translation(0)
    print("Camera 0 extrinsics loaded from file.")

    # Initialize dlib's face detector and shape predictor.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Hard–set the 3D rear window point (in homogeneous coordinates).
    threeD_window_pt = [
        [-5e+05],
        [8.65304694e+04],
        [-1.1e+04],
        [1.00000000e+00]
    ]
    # Convert to a 3D tuple (dropping the homogeneous coordinate).
    window_pt = (
        threeD_window_pt[0][0],
        threeD_window_pt[1][0],
        threeD_window_pt[2][0]
    )

    # Initialize serial communication.
    ser = serial.Serial("COM4", 115200, timeout=0.5)

    while True:
        # Capture frames from both cameras.
        frame_left = picam2_left.capture_array()
        frame_right = picam2_right.capture_array()

        if frame_left is None or frame_right is None:
            print("No frame captured; exiting.")
            break

        # Convert frames to grayscale for face detection.
        gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

        # Detect faces.
        faces_left = detector(gray_left)
        faces_right = detector(gray_right)

        if len(faces_left) > 0 and len(faces_right) > 0:
            # Use the first detected face from each camera.
            shape_left = predictor(gray_left, faces_left[0])
            shape_right = predictor(gray_right, faces_right[0])

            # Use the 28th landmark (index 27) as the mid-face point.
            midpt_left = (shape_left.part(27).x, shape_left.part(27).y)
            midpt_right = (shape_right.part(27).x, shape_right.part(27).y)

            # Draw the detected landmark.
            cv.circle(frame_left, midpt_left, radius=2, color=(0, 255, 0), thickness=2)
            cv.circle(frame_right, midpt_right, radius=2, color=(0, 255, 0), thickness=2)
            print("Left landmark:", midpt_left)
            print("Right landmark:", midpt_right)

            # Compute the 3D point using DLT.
            threeD_midpt = DLT(P_left, P_right, midpt_left, midpt_right)
            if len(threeD_midpt) == 4:
                threeD_midpt = [threeD_midpt[i] / threeD_midpt[3] for i in range(3)]
            print("3D midpoint:", threeD_midpt)

            try:
                # --------------------------------------
                # Transform the face point & window point
                # into the mirror's coordinate system.
                # --------------------------------------
                # We treat the mirror as the origin, so we subtract
                # (d1, d2, 0) from the camera-based 3D coordinates.
                face_pt_mirror = (
                    threeD_midpt[0] - d1,
                    threeD_midpt[1] - d2,
                    threeD_midpt[2]
                )
                window_pt_mirror = (
                    window_pt[0] - d1,
                    window_pt[1] - d2,
                    window_pt[2]
                )

                # --------------------------------------
                # Plane definitions in the mirror frame
                # --------------------------------------

                # Construct the "driver plane" from:
                # (0,0,0), (faceX,0,faceZ), (faceX,faceY,faceZ)
                driver_plane = Plane(
                    Point3D(0, 0, 0),
                    Point3D(face_pt_mirror[0], 0, face_pt_mirror[2]),
                    Point3D(face_pt_mirror[0], face_pt_mirror[1], face_pt_mirror[2])
                )

                # Construct the "rear window plane" similarly.
                rear_window_plane = Plane(
                    Point3D(0, 0, 0),
                    Point3D(window_pt_mirror[0], 0, window_pt_mirror[2]),
                    Point3D(window_pt_mirror[0], window_pt_mirror[1], window_pt_mirror[2])
                )

                # Reference plane: xy–plane.
                xy_plane = Plane(
                    Point3D(0, 0, 0),
                    Point3D(1, 0, 0),
                    Point3D(0, 1, 0)
                )
                # Reference plane: xz–plane.
                xz_plane = Plane(
                    Point3D(0, 0, 0),
                    Point3D(1, 0, 0),
                    Point3D(0, 0, 1)
                )

                # Compute angles:
                alpha = driver_plane.angle_between(rear_window_plane)
                beta = driver_plane.angle_between(xy_plane)
                # Heuristically combine angles to compute yaw.
                # The division by 2 is from the law of reflection/bisector logic.
                yaw = math.degrees(alpha / 2 + beta)
                print("Heuristically computed yaw:", yaw)

                # Construct the "driver mirror plane" using
                # (0,0,0), the face point, and the window point (all in mirror frame).
                driver_mirror_plane = Plane(
                    Point3D(0, 0, 0),
                    Point3D(face_pt_mirror[0], face_pt_mirror[1], face_pt_mirror[2]),
                    Point3D(window_pt_mirror[0], window_pt_mirror[1], window_pt_mirror[2])
                )
                # Compute pitch as the angle between the driver mirror plane and the xz–plane.
                pitch = math.degrees(driver_mirror_plane.angle_between(xz_plane))
                print("Computed pitch:", pitch)

                # Send the angles over serial (each followed by a newline).
                ser.write(f"{yaw}\n".encode())
                ser.write(f"{pitch}\n".encode())

            except Exception as e:
                print("Error computing angles:", e)
        else:
            print("Face not detected in one or both cameras.")

        # Show live frames.
        cv.imshow('Left Camera', frame_left)
        cv.imshow('Right Camera', frame_right)

        # Exit if ESC is pressed.
        if cv.waitKey(1) & 0xFF == 27:
            break

    # Clean up.
    cv.destroyAllWindows()
    picam2_left.stop()
    picam2_right.stop()
    ser.close()

if __name__ == '__main__':
    main()
