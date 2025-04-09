import numpy as np
import cv2 as cv
import dlib
import math
from time import sleep
from picamera2 import Picamera2
from utils import DLT, get_projection_matrix, read_rotation_translation

# Helper function to extract Euler angles from a rotation matrix.
def rotationMatrixToEulerAngles(R):
    """
    Computes Euler angles (roll, pitch, yaw) from a 3x3 rotation matrix R.
    Assumes the rotation order is: roll (X axis), pitch (Y axis), yaw (Z axis).
    Returns angles in radians.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0

    return roll, pitch, yaw

def main():
    # Define desired frame resolution (height, width)
    frame_shape = [1232, 1640]

    # Initialize picamera2 objects for the stereo cameras.
    # Adjust camera indices as needed.
    picam2_left = Picamera2(camera_num=0)
    config_left = picam2_left.create_preview_configuration(main={"size": (frame_shape[1], frame_shape[0])})
    picam2_left.configure(config_left)
    picam2_left.start()

    picam2_right = Picamera2(camera_num=1)
    config_right = picam2_right.create_preview_configuration(main={"size": (frame_shape[1], frame_shape[0])})
    picam2_right.configure(config_right)
    picam2_right.start()

    # Retrieve projection matrices from your pre-computed calibration files.
    P_left = get_projection_matrix(0)
    P_right = get_projection_matrix(1)

    # Read the rotation (and translation) from calibration files for camera 0.
    R0, t0 = read_rotation_translation(0)
    # Convert R0 to Euler angles (in radians) then to degrees.
    roll0, pitch0, yaw0 = rotationMatrixToEulerAngles(R0)
    print("Camera 0 Extrinsics (from rot_trans_c0.dat):")
    print("Roll: {:.2f} deg, Pitch: {:.2f} deg, Yaw: {:.2f} deg".format(
        math.degrees(roll0), math.degrees(pitch0), math.degrees(yaw0)
    ))

    # Initialize dlib's face detector and the 68-point shape predictor.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Hard-set 3D rear window point (in homogeneous coordinates).
    # This value is fixed based on your setup.
    threeD_window_pt = [[ -5e+05],
                        [ 8.65304694e+04],
                        [-1.1e+04],
                        [ 1.00000000e+00]]
    # Convert to a 3D point tuple (drop the homogeneous coordinate)
    window_pt = (threeD_window_pt[0][0], threeD_window_pt[1][0], threeD_window_pt[2][0])

    while True:
        # Capture frames from each camera
        frame_left = picam2_left.capture_array()
        frame_right = picam2_right.capture_array()

        if frame_left is None or frame_right is None:
            print("No frame captured; exiting.")
            break

        # Convert to grayscale for face detection
        gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

        # Detect faces in both images.
        faces_left = detector(gray_left)
        faces_right = detector(gray_right)

        if len(faces_left) > 0 and len(faces_right) > 0:
            # For simplicity, use the first detected face from each camera.
            shape_left = predictor(gray_left, faces_left[0])
            shape_right = predictor(gray_right, faces_right[0])

            # Choose the 28th landmark (index 27) as the mid-face point.
            midpt_left = (shape_left.part(27).x, shape_left.part(27).y)
            midpt_right = (shape_right.part(27).x, shape_right.part(27).y)

            # Draw the landmark on each image.
            cv.circle(frame_left, midpt_left, radius=2, color=(0, 255, 0), thickness=2)
            cv.circle(frame_right, midpt_right, radius=2, color=(0, 255, 0), thickness=2)

            print("Left landmark:", midpt_left)
            print("Right landmark:", midpt_right)

            # Compute the 3D position from the two 2D points using DLT.
            threeD_midpt = DLT(P_left, P_right, midpt_left, midpt_right)
            # Normalize if returned as a homogeneous vector (if length 4).
            if len(threeD_midpt) == 4:
                threeD_midpt = [threeD_midpt[i] / threeD_midpt[3] for i in range(3)]
            print("3D midpoint:", threeD_midpt)

            # --- Method 1: Compute angles from the difference vector ---
            # Compute vector from the driver's 3D point to the fixed window point.
            diff_vec = np.array(window_pt) - np.array(threeD_midpt)
            # Assume that the Z component is the forward direction.
            yaw_infer  = math.degrees(math.atan2(diff_vec[0], diff_vec[2]))
            pitch_infer = math.degrees(math.atan2(diff_vec[1], diff_vec[2]))
            print("Inferred angles from 3D point difference:")
            print("Pitch: {:.2f} deg, Yaw: {:.2f} deg".format(pitch_infer, yaw_infer))
        else:
            print("Face not detected in one or both cameras.")

        # Show the frames.
        cv.imshow('Left Camera', frame_left)
        cv.imshow('Right Camera', frame_right)

        # Exit on pressing the ESC key.
        if cv.waitKey(1) & 0xFF == 27:
            break

    # Clean up.
    cv.destroyAllWindows()
    picam2_left.stop()
    picam2_right.stop()

if __name__ == '__main__':
    main()

