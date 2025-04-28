#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import sys
import yaml
import os
import time
from picamera2 import Picamera2

# Global dictionary for calibration settings.
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

def load_extrinsics(filename):
    if not os.path.exists(filename):
        print("Extrinsic parameters file does not exist:", filename)
        quit()
    with open(filename, 'r') as f:
        lines = f.readlines()
    try:
        R_index = lines.index("R:\n")
        T_index = lines.index("T:\n")
    except ValueError:
        print("Error parsing extrinsics in file", filename)
        quit()
    R_lines = lines[R_index+1 : T_index]
    R = [ [float(x) for x in line.split()] for line in R_lines if line.strip() != "" ]
    R = np.array(R)
    T_lines = lines[T_index+1:]
    T = [ [float(x) for x in line.split()] for line in T_lines if line.strip() != "" ]
    T = np.array(T)
    return R, T

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P

def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P

def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift=50.):
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])
    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)
    coordinate_points = np.array([[0., 0., 0.],
                                  [1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]])
    z_shift = np.array([0., 0., _zshift]).reshape((1, 3))
    draw_axes_points = 5 * coordinate_points + z_shift
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera0.append(uv)
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera1.append(uv)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cam0_index = calibration_settings['camera0']
    cam1_index = calibration_settings['camera1']

    time.sleep(3)
    picam2_cam0 = Picamera2(camera_num=cam0_index)
    config0 = picam2_cam0.create_preview_configuration(main={"size": (width, height)})
    picam2_cam0.configure(config0)
    picam2_cam0.start()

    picam2_cam1 = Picamera2(camera_num=cam1_index)
    config1 = picam2_cam1.create_preview_configuration(main={"size": (width, height)})
    picam2_cam1.configure(config1)
    picam2_cam1.start()

    while True:
        frame0 = picam2_cam0.capture_array()
        frame1 = picam2_cam1.capture_array()
        if frame0 is None or frame1 is None:
            print('Video stream not returning frame data')
            picam2_cam0.stop()
            picam2_cam1.stop()
            del picam2_cam0, picam2_cam1
            time.sleep(3)
            quit()

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)
        k = cv.waitKey(1)
        if k == 27:
            picam2_cam0.stop()
            picam2_cam1.stop()
            del picam2_cam0, picam2_cam1
            break

    cv.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 check_calibration.py calibration_settings.yaml')
        quit()
    
    # Load calibration settings
    parse_calibration_settings_file(sys.argv[1])
    
    # Load intrinsic parameters from file
    cmtx0, dist0 = load_camera_intrinsics('camera0')
    cmtx1, dist1 = load_camera_intrinsics('camera1')
    
    # Load extrinsic calibration parameters saved from File 2
    extrinsics_cam0_file = os.path.join('camera_parameters', 'camera0_rot_trans.dat')
    extrinsics_cam1_file = os.path.join('camera_parameters', 'camera1_rot_trans.dat')
    R0, T0 = load_extrinsics(extrinsics_cam0_file)
    R1, T1 = load_extrinsics(extrinsics_cam1_file)
    
    # Prepare camera data for visualization
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    
    print("Starting calibration check. Press ESC to exit the visualization windows.")
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift=60.)
