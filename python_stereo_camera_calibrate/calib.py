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

# This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

# Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2, :] - P1[1, :],
         P1[0, :] - point1[0]*P1[2, :],
         point2[1]*P2[2, :] - P2[1, :],
         P2[0, :] - point2[0]*P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]

# Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    global calibration_settings
    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    print('Using for calibration settings:', filename)
    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)
    # Rudimentary check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()

# Open camera stream using picamera2 and save frames (single camera)
def save_frames_single_camera(camera_name):
    # Create frames directory
    if not os.path.exists('frames'):
        os.mkdir('frames')

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # Look up the camera index from settings (e.g. 0 or 1)
    camera_index = calibration_settings[camera_name]
    # Initialize picamera2 with specific camera index
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

# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
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
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)
    return cmtx, dist

# Save camera intrinsic parameters to file.
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n')
        for l in camera_matrix:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.write('distortion:\n')
        for en in distortion_coefs[0]:
            outf.write(str(en) + ' ')
        outf.write('\n')

# Open paired calibration frames using picamera2 and save them simultaneously.
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

    # Add a longer delay to ensure previous resources are released.
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

# Open paired calibration frames and stereo calibrate for cam0 to cam1 coordinate transformations.
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
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []
    objpoints = []  # 3d points in real world space

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
    print('rmse:', ret)
    cv.destroyAllWindows()
    return R, T

# Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix.
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P

# Turn camera calibration data into projection matrix.
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly using picamera2.
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

def get_world_space_origin(cmtx, dist, img_path):
    frame = cv.imread(img_path, 1)
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
    cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50, 50),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)
    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _ = cv.Rodrigues(rvec)
    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):
    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)
    unitv_points = 5 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32').reshape((4, 1, 3))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)
    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)
    return R_W1, T_W1

def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    with open(camera0_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.write('T:\n')
        for l in T0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    with open(camera1_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R1:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.write('T:\n')
        for en in T1:
            outf.write(str(en) + ' ')
        outf.write('\n')
    return R0, T0, R1, T1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()
    
    # Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])

    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera('camera0')  # Save frames for camera0
    save_frames_single_camera('camera1')  # Save frames for camera1

    """Step2. Obtain camera intrinsic matrices and save them"""
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx0, dist0, 'camera0')
    images_prefix = os.path.join('frames', 'camera1*')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx1, dist1, 'camera1')

    """Step3. Save calibration frames for both cameras simultaneously"""
    save_frames_two_cams('camera0', 'camera1')

    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)

    """Step5. Save calibration data where camera0 defines the world space origin."""
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    save_extrinsic_calibration_parameters(R0, T0, R, T)
    R1 = R
    T1 = T
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift=60.)

    # Optional: Define a different origin point and save the calibration data
    # R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, os.path.join('frames_pair', 'camera0_4.png'))
    # R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
    #                                           cmtx1, dist1, R1, T1,
    #                                           os.path.join('frames_pair', 'camera0_4.png'),
    #                                           os.path.join('frames_pair', 'camera1_4.png'))
    # save_extrinsic_calibration_parameters(R_W0, T_W0, R_W1, T_W1, prefix='world_to_')
