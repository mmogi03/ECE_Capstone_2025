import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import time
from picamera2 import Picamera2

# Global dictionary for calibration settings
calibration_settings = {}

# Given projection matrices P1 and P2 and corresponding image points, return triangulated 3D point.
def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2, :] - P1[1, :],
         P1[0, :] - point1[0]*P1[2, :],
         point2[1]*P2[2, :] - P2[1, :],
         P2[0, :] - point2[0]*P2[2, :]
         ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]

# Parse the calibration settings YAML file.
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

# Capture images from one camera using Picamera2.
def save_frames_single_camera(camera_name):
    if not os.path.exists("frames"):
        os.mkdir("frames")
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    number_to_save = calibration_settings["mono_calibration_frames"]
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]
    camera_index = calibration_settings[camera_name]
    picam2 = Picamera2(camera_num=camera_index)
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()
    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True:
        frame = picam2.capture_array()
        if frame is None:
            print("No video data received from camera. Exiting...")
            picam2.stop()
            del picam2
            time.sleep(3)
            quit()
        frame_small = cv.resize(frame, None, fx=1/view_resize, fy=1/view_resize)
        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50,50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            if cooldown <= 0:
                savename = os.path.join("frames", f"{camera_name}_{saved_count}.png")
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time
        cv.imshow("frame_small", frame_small)
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

# Calibrate a single camera using saved images.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    images_names = glob.glob(images_prefix)
    images = [cv.imread(imname, 1) for imname in images_names]
    criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    world_scaling = calibration_settings["checkerboard_box_size_scale"]
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)
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
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1,-1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25,25),
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

def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")
    out_filename = os.path.join("camera_parameters", f"{camera_name}_intrinsics.dat")
    with open(out_filename, "w") as outf:
        outf.write("intrinsic:\n")
        for l in camera_matrix:
            outf.write(" ".join(str(x) for x in l) + "\n")
        outf.write("distortion:\n")
        outf.write(" ".join(str(x) for x in distortion_coefs[0]) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 intrinsics_calibration.py calibration_settings.yaml")
        quit()
    parse_calibration_settings_file(sys.argv[1])
    save_frames_single_camera("camera0")
    save_frames_single_camera("camera1")
    images_prefix = os.path.join("frames", "camera0*")
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx0, dist0, "camera0")
    images_prefix = os.path.join("frames", "camera1*")
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx1, dist1, "camera1")
