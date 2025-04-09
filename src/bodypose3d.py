import cv2 as cv
import dlib
import numpy as np
from picamera2 import Picamera2
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

# Desired frame resolution: height x width
frame_shape = [1232, 1640]  # (height=1232, width=1640)

def run_dlib(camera_index0, camera_index1, P0, P1, save_interval=10):
    # Initialize picamera2 objects for each camera
    picam2_0 = Picamera2(camera_num=camera_index0)
    config0 = picam2_0.create_preview_configuration(main={"size": (frame_shape[1], frame_shape[0])})
    picam2_0.configure(config0)
    picam2_0.start()

    picam2_1 = Picamera2(camera_num=camera_index1)
    config1 = picam2_1.create_preview_configuration(main={"size": (frame_shape[1], frame_shape[0])})
    picam2_1.configure(config1)
    picam2_1.start()

    # Initialize dlib face detector and the 68 facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Containers for detected keypoints per frame
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    frame_count = 0

    while True:
        # Capture frames from each camera
        frame0 = picam2_0.capture_array()
        frame1 = picam2_1.capture_array()

        if frame0 is None or frame1 is None:
            break

        # Crop to a square (if needed) based on frame_shape.
        if frame0.shape[1] != frame_shape[0]:
            start_col = frame_shape[1] // 2 - frame_shape[0] // 2
            end_col = frame_shape[1] // 2 + frame_shape[0] // 2
            frame0 = frame0[:, start_col:end_col]
            frame1 = frame1[:, start_col:end_col]

        # Convert to grayscale for dlib detection
        gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        # Detect faces in each frame
        rects0 = detector(gray0, 1)
        rects1 = detector(gray1, 1)

        # Process camera 0: if a face is detected, extract its 68 landmarks.
        frame0_keypoints = []
        if len(rects0) > 0:
            shape0 = predictor(gray0, rects0[0])
            for i in range(68):
                x, y = shape0.part(i).x, shape0.part(i).y
                frame0_keypoints.append([x, y])
                cv.circle(frame0, (x, y), 2, (0, 0, 255), -1)
        else:
            # If no face is detected, fill with dummy keypoints.
            frame0_keypoints = [[-1, -1]] * 68
        kpts_cam0.append(frame0_keypoints)

        # Process camera 1 similarly.
        frame1_keypoints = []
        if len(rects1) > 0:
            shape1 = predictor(gray1, rects1[0])
            for i in range(68):
                x, y = shape1.part(i).x, shape1.part(i).y
                frame1_keypoints.append([x, y])
                cv.circle(frame1, (x, y), 2, (0, 0, 255), -1)
        else:
            frame1_keypoints = [[-1, -1]] * 68
        kpts_cam1.append(frame1_keypoints)

        # Compute the 3D positions for each landmark using DLT.
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                p3d = [-1, -1, -1]
            else:
                p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(p3d)
        kpts_3d.append(frame_p3ds)

        # Show the annotated frames
        cv.imshow('cam0', frame0)
        cv.imshow('cam1', frame1)
        k = cv.waitKey(1)
        if k & 0xFF == 27:  # ESC key to exit
            break

        frame_count += 1

        # Save keypoints to disk every `save_interval` frames.
        if frame_count % save_interval == 0:
            write_keypoints_to_disk('kpts_cam0.dat', np.array(kpts_cam0))
            write_keypoints_to_disk('kpts_cam1.dat', np.array(kpts_cam1))
            write_keypoints_to_disk('kpts_3d.dat', np.array(kpts_3d))
    
    cv.destroyAllWindows()
    picam2_0.stop()
    picam2_1.stop()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':
    # Use camera indices 0 and 1 for picamera2
    camera_index0 = 0
    camera_index1 = 1

    # Get projection matrices for each camera from utils.
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    # Run dlib facial landmark detection and compute corresponding 3D positions.
    kpts_cam0, kpts_cam1, kpts_3d = run_dlib(camera_index0, camera_index1, P0, P1, save_interval=10)

    # Final save to ensure all data is written.
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)
