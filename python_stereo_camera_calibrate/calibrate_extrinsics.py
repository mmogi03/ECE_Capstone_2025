#!/usr/bin/env python3
import cv2 as cv
import glob
import numpy as np
import sys
import yaml
import os
import time
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
        print('camera0 key was not found in the settings file.')
        quit()

def load_camera_intrinsics(camera_name):
    filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    if not os.path.exists(filename):
        print("Intrinsic parameters file does not exist:", filename)
        quit()
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]
    try:
        intrinsic_index = lines.index("intrinsic:")
        distortion_index = lines.index("distortion:")
    except ValueError:
        print("Error reading intrinsic file:", filename)
        quit()
    intrinsic_lines = lines[intrinsic_index+1:distortion_index]
    intrinsic = [ [float(x) for x in line.split()] for line in intrinsic_lines ]
    intrinsic = np.array(intrinsic)
    distortion_line = lines[distortion_index+1]
    distortion = [float(x) for x in distortion_line.split()]
    distortion = np.array([distortion])
    return intrinsic, distortion

def save_frames_two_cams(camera0_name, camera1_name):
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    number_to_save = calibration_settings['stereo_calibration_frames']
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']

    camera0_index = calibration_settings[camera0_name]
    camera1_index = calibration_settings[camera1_name]

    time.sleep(3)

    picam2_cam0 = Picamera2(camera_num=camera0_index)
    config0 = picam2_cam0.create_preview_configuration(main={"size": (width, height)})
    picam2_cam0.configure(config0)
    picam2_cam0.start()

    picam2_cam1 = Picamera2(camera_num=camera1_index)
    config1 = picam2_cam1.create_preview_configuration(main={"size": (width, height)})
    picam2_cam1.configure(config1)
    picam2_cam1.start()

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        frame0 = picam2_cam0.capture_array()
        frame1 = picam2_cam1.capture_array()
        if frame0 is None or frame1 is None:
            print('Cameras not returning video data. Exiting...')
            picam2_cam0.stop()
            picam2_cam1.stop()
            del picam2_cam0, picam2_cam1
            time.sleep(3)
            quit()

        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50, 100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50, 100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50, 100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            if cooldown <= 0:
                savename0 = os.path.join('frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename0, frame0)
                savename1 = os.path.join('frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename1, frame1)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)
        if k == 27:
            picam2_cam0.stop()
            picam2_cam1.stop()
            del picam2_cam0, picam2_cam1
            cv.destroyAllWindows()
            time.sleep(3)
            quit()
        if k == 32:
            start = True
        if saved_count == number_to_save:
            break

    picam2_cam0.stop()
    picam2_cam1.stop()
    del picam2_cam0, picam2_cam1
    cv.destroyAllWindows()
    time.sleep(3)

def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]
    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv.imshow('img', frame0)
            cv.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)
            if k & 0xFF == ord('s'):
                print('skipping')
                continue
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx0, dist0, mtx1, dist1,
        (width, height), criteria=criteria,
        flags=stereocalibration_flags
    )
    print('Stereo Calibration RMSE:', ret)
    cv.destroyAllWindows()
    return R, T

def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    with open(camera0_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for row in R0:
            outf.write(' '.join([str(val) for val in row]) + '\n')
        outf.write('T:\n')
        for row in T0:
            outf.write(' '.join([str(val) for val in row]) + '\n')
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    with open(camera1_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for row in R1:
            outf.write(' '.join([str(val) for val in row]) + '\n')
        outf.write('T:\n')
        for val in T1.flatten():
            outf.write(str(val) + ' ')
        outf.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 calibrate_extrinsics.py calibration_settings.yaml')
        quit()
    
    # Load calibration settings
    parse_calibration_settings_file(sys.argv[1])
    
    # Load intrinsic parameters saved from File 1
    cmtx0, dist0 = load_camera_intrinsics('camera0')
    cmtx1, dist1 = load_camera_intrinsics('camera1')
    
    # --- Step 3: Capture paired frames ---
    print("Starting paired frame capture for stereo calibration...")
    save_frames_two_cams('camera0', 'camera1')
    
    # --- Step 4: Stereo calibration using paired frames ---
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    print("Performing stereo calibration...")
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)
    
    # Save extrinsic calibration parameters (with camera0 as the world origin)
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    save_extrinsic_calibration_parameters(R0, T0, R, T)
    
    print("Extrinsic calibration complete. Calibration parameters saved in 'camera_parameters'.")
