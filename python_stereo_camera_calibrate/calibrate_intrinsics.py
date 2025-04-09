#!/usr/bin/env python3
import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import time
# Import picamera2 for camera control
from picamera2 import Picamera2

# Global dictionary to store calibration settings loaded from YAML.
calibration_settings = {}

def parse_calibration_settings_file(filename):
    global calibration_settings
    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    print('Using calibration settings file:', filename)
    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()

def save_frames_single_camera(camera_name):
    # Create a directory to save frames if it does not exist
    if not os.path.exists('frames'):
        os.mkdir('frames')

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # Look up the camera index from settings (e.g. 0 or 1)
    camera_index = calibration_settings[camera_name]
    # Initialize Picamera2 with the specific camera index
    picam2 = Picamera2(camera_num=camera_index)
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        frame = picam2.capture_array()  # returns a NumPy array
        if frame is None:
            print("No video data received from camera. Exiting...")
            picam2.stop()
            del picam2
            time.sleep(3)
            quit()

        frame_small = cv.resize(frame, None, fx=1/view_resize, fy=1/view_resize)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50, 100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)
        if k == 27:
            picam2.stop()
            del picam2
            cv.destroyAllWindows()
            time.sleep(3)
            quit()
        if k == 32:
            start = True
        if saved_count == number_to_save:
            break

    picam2.stop()
    del picam2
    cv.destroyAllWindows()
    time.sleep(3)

def calibrate_camera_for_intrinsic_parameters(images_prefix):
    images_names = glob.glob(images_prefix)
    images = [cv.imread(imname, 1) for imname in images_names]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp
    width = images[0].shape[1]
    height = images[0].shape[0]
    imgpoints = []  # 2d points in image plane.
    objpoints = []  # 3d points in real world space

    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ret:
            conv_size = (11, 11)
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.imshow('img', frame)
            k = cv.waitKey(0)
            if k & 0xFF == ord('s'):
                print('skipping')
                continue
            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('Calibration RMSE:', ret)
    print('Camera matrix:\n', cmtx)
    print('Distortion coefficients:', dist)
    return cmtx, dist

def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n')
        for row in camera_matrix:
            outf.write(' '.join([str(val) for val in row]) + '\n')
        outf.write('distortion:\n')
        outf.write(' '.join([str(val) for val in distortion_coefs[0]]) + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 calibrate_intrinsics.py calibration_settings.yaml')
        quit()
    
    # Load calibration settings
    parse_calibration_settings_file(sys.argv[1])
    
    # --- Step 1: Capture single-camera frames ---
    print("Starting single camera frame capture for camera0...")
    save_frames_single_camera('camera0')
    print("Starting single camera frame capture for camera1...")
    save_frames_single_camera('camera1')
    
    # --- Step 2: Calibrate each camera for intrinsic parameters ---
    images_prefix = os.path.join('frames', 'camera0*')
    print("Calibrating intrinsic parameters for camera0...")
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx0, dist0, 'camera0')
    
    images_prefix = os.path.join('frames', 'camera1*')
    print("Calibrating intrinsic parameters for camera1...")
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx1, dist1, 'camera1')
    
    print("Intrinsic calibration complete.")
