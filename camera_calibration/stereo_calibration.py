import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import time
from picamera2 import Picamera2

calibration_settings = {}

# Parse calibration settings.
def parse_calibration_settings_file(filename):
    global calibration_settings
    if not os.path.exists(filename):
        print("File does not exist:", filename)
        quit()
    print("Using for calibration settings:", filename)
    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)
    if "camera0" not in calibration_settings.keys():
        print("camera0 key not found in settings file.")
        quit()

# (Re)define calibrate_camera_for_intrinsic_parameters so we can recalibrate intrinsics if needed.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    images_names = glob.glob(images_prefix)
    images = [cv.imread(imname, 1) for imname in images_names]
    criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp
    width = images[0].shape[1]
    height = images[0].shape[0]
    imgpoints = []
    objpoints = []
    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ret:
            conv_size = (11,11)
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.putText(frame, 'Press "s" to skip sample if needed', (25,25),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.imshow("img", frame)
            k = cv.waitKey(0)
            if k & 0xFF == ord("s"):
                print("skipping")
                continue
            objpoints.append(objp)
            imgpoints.append(corners)
    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print("rmse:", ret)
    print("camera matrix:\n", cmtx)
    print("distortion coeffs:", dist)
    return cmtx, dist

# Capture paired frames from both cameras.
def save_frames_two_cams(camera0_name, camera1_name):
    if not os.path.exists("frames_pair"):
        os.mkdir("frames_pair")
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]
    number_to_save = calibration_settings["stereo_calibration_frames"]
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
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
            print("Cameras not returning video data. Exiting...")
            picam2_cam0.stop()
            picam2_cam1.stop()
            del picam2_cam0, picam2_cam1
            time.sleep(3)
            quit()
        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)
        if not start:
            cv.putText(frame0_small, "Make sure both cameras see the calibration pattern", (50,50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50,100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50,50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50,100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50,50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50,100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            if cooldown <= 0:
                savename0 = os.path.join("frames_pair", f"{camera0_name}_{saved_count}.png")
                cv.imwrite(savename0, frame0)
                savename1 = os.path.join("frames_pair", f"{camera1_name}_{saved_count}.png")
                cv.imwrite(savename1, frame1)
                saved_count += 1
                cooldown = cooldown_time
        cv.imshow("frame0_small", frame0_small)
        cv.imshow("frame1_small", frame1_small)
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

# Perform stereo calibration using the paired images.
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]
    criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)
    objp = world_scaling * objp
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]
    imgpoints_left = []
    imgpoints_right = []
    objpoints = []
    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
        if ret1 and ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)
            cv.drawChessboardCorners(frame0, (rows, columns), corners1, ret1)
            cv.imshow("img", frame0)
            cv.drawChessboardCorners(frame1, (rows, columns), corners2, ret2)
            cv.imshow("img2", frame1)
            k = cv.waitKey(0)
            if k & 0xFF == ord("s"):
                print("skipping")
                continue
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx0, dist0, mtx1, dist1, (width, height),
        criteria=criteria, flags=stereocalibration_flags
    )
    print("rmse:", ret)
    cv.destroyAllWindows()
    return R, T

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3,3] = t.reshape(3)
    P[3,3] = 1
    return P

def get_projection_matrix(cmtx, R, T):
    return cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]

# Display a live overlay for calibration by combining the two views.
def check_calibration(camera0_data, camera1_data, _zshift=50.):
    cmtx0, dist0, R0, T0 = camera0_data
    cmtx1, dist1, R1, T1 = camera1_data
    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1,3))
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
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    cam0_index = calibration_settings["camera0"]
    cam1_index = calibration_settings["camera1"]
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
            print("Video stream not returning frame data")
            picam2_cam0.stop()
            picam2_cam1.stop()
            del picam2_cam0, picam2_cam1
            time.sleep(3)
            quit()
        # Option: resize frames and combine side-by-side.
        scale_factor = 0.5
        small0 = cv.resize(frame0, None, fx=scale_factor, fy=scale_factor)
        small1 = cv.resize(frame1, None, fx=scale_factor, fy=scale_factor)
        combined = np.hstack((small0, small1))
        # Draw axes on each frame:
        for col, pt in zip([(0,0,255), (0,255,0), (255,0,0)], pixel_points_camera0[1:]):
            cv.line(frame0, tuple(pixel_points_camera0[0].astype(np.int32)), tuple(pt.astype(np.int32)), col, 2)
        for col, pt in zip([(0,0,255), (0,255,0), (255,0,0)], pixel_points_camera1[1:]):
            cv.line(frame1, tuple(pixel_points_camera1[0].astype(np.int32)), tuple(pt.astype(np.int32)), col, 2)
        # Alternatively, combine the resized views:
        cv.imshow("Stereo Calibration", combined)
        k = cv.waitKey(1)
        if k == 27:
            picam2_cam0.stop()
            picam2_cam1.stop()
            del picam2_cam0, picam2_cam1
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 stereo_calibration.py calibration_settings.yaml")
        quit()
    parse_calibration_settings_file(sys.argv[1])
    # Step 3: Capture paired stereo frames.
    save_frames_two_cams("camera0", "camera1")
    # Step 4: Perform stereo calibration.
    frames_prefix_c0 = os.path.join("frames_pair", "camera0*")
    frames_prefix_c1 = os.path.join("frames_pair", "camera1*")
    # (Optionally, recalibrate intrinsics from the single-camera images.)
    images_prefix = os.path.join("frames", "camera0*")
    cmtx0, dist0 = None, None
    if len(glob.glob(images_prefix)) > 0:
        cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    images_prefix = os.path.join("frames", "camera1*")
    cmtx1, dist1 = None, None
    if len(glob.glob(images_prefix)) > 0:
        cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)
    # Step 5: Save extrinsic calibration (using camera0 as the world origin).
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0.,0.,0.]).reshape((3,1))
    out_file0 = os.path.join("camera_parameters", "camera0_rot_trans.dat")
    out_file1 = os.path.join("camera_parameters", "camera1_rot_trans.dat")
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")
    with open(out_file0, "w") as f:
        f.write("R:\n" + "\n".join(" ".join(str(x) for x in row) for row in R0) + "\n")
        f.write("T:\n" + " ".join(str(x) for x in T0.flatten()) + "\n")
    with open(out_file1, "w") as f:
        f.write("R:\n" + "\n".join(" ".join(str(x) for x in row) for row in R) + "\n")
        f.write("T:\n" + " ".join(str(x) for x in T.flatten()) + "\n")
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R, T]
    # Step 6: Check calibration overlay.
    check_calibration(camera0_data, camera1_data, _zshift=60.)
