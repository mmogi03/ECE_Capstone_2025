#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import os
import glob
import argparse
from picamera2 import Picamera2

def capture_images(num_images=20, 
                   save_dir_left='images/stereoLeft', 
                   save_dir_right='images/stereoRight'):
    """
    Capture stereo images using the IMX219-83 with Picamera2.
    Two independent Picamera2 objects are used (one for each camera).
    Press 's' to save the current stereo pair, or 'ESC' to exit.
    """
    os.makedirs(save_dir_left, exist_ok=True)
    os.makedirs(save_dir_right, exist_ok=True)
    
    # Initialize and configure cameras
    picamLeft = Picamera2(camera_num=0)
    picamLeft.configure(picamLeft.create_preview_configuration())
    picamLeft.start()
    
    picamRight = Picamera2(camera_num=1)
    picamRight.configure(picamRight.create_preview_configuration())
    picamRight.start()
    
    print("Press 's' to save stereo image pair, or 'ESC' to exit.")
    count = 0
    while count < num_images:
        # Capture frames from both cameras
        frameLeft = picamLeft.capture_array()
        frameRight = picamRight.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        imgL = cv.cvtColor(frameLeft, cv.COLOR_RGB2BGR)
        imgR = cv.cvtColor(frameRight, cv.COLOR_RGB2BGR)
        
        # Display the images
        cv.imshow("Left Camera", imgL)
        cv.imshow("Right Camera", imgR)
        
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('s'):
            left_filename = os.path.join(save_dir_left, f"imageL{count:02d}.png")
            right_filename = os.path.join(save_dir_right, f"imageR{count:02d}.png")
            cv.imwrite(left_filename, imgL)
            cv.imwrite(right_filename, imgR)
            print(f"Saved stereo pair {count}")
            count += 1

    cv.destroyAllWindows()
    picamLeft.stop()
    picamRight.stop()
    print("Image capture complete.")

def stereo_calibration(chessboard_size=(8, 6),
                       square_size=30,
                       image_dir_left='images/stereoLeft',
                       image_dir_right='images/stereoRight',
                       output_dir='calibration_params',
                       frame_size=(640, 480)):
    """
    Perform stereo calibration using captured chessboard images.
    Detects chessboard corners in both left and right images, calibrates each
    camera individually, then runs stereo calibration to compute rotation and
    translation between the two cameras.
    Calibration parameters are saved as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Termination criteria for subpixel corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points: (0,0,0), (1,0,0), ... scaled by the actual square size (in mm)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    objpoints = []   # 3D points in real-world space
    imgpointsL = []  # 2D points in left image plane
    imgpointsR = []  # 2D points in right image plane

    images_left = sorted(glob.glob(os.path.join(image_dir_left, '*.png')))
    images_right = sorted(glob.glob(os.path.join(image_dir_right, '*.png')))

    if len(images_left) != len(images_right):
        print("The number of left and right images do not match!")
        return

    for img_left_path, img_right_path in zip(images_left, images_right):
        imgL = cv.imread(img_left_path)
        imgR = cv.imread(img_right_path)
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        retL, cornersL = cv.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR:
            objpoints.append(objp)
            
            cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)
            
            cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)
            
            # Optionally display detected corners for verification
            cv.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
            cv.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)
            cv.imshow("Left Chessboard", imgL)
            cv.imshow("Right Chessboard", imgR)
            cv.waitKey(500)
    cv.destroyAllWindows()

    # Calibrate each camera individually
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objpoints, imgpointsL, frame_size, None, None)
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objpoints, imgpointsR, frame_size, None, None)
    
    # Stereo calibration: fix the intrinsic parameters and compute the extrinsics
    flags = cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 1e-6)
    retStereo, cameraMatrixL, distL, cameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        cameraMatrixL, distL, cameraMatrixR, distR,
        frame_size, criteria=criteria_stereo, flags=flags)
    
    # Stereo Rectification (0 to crop, 1 to keep full images)
    rectify_scale = 1
    R1, R2, projMatrixL, projMatrixR, Q, roiL, roiR = cv.stereoRectify(
        cameraMatrixL, distL, cameraMatrixR, distR,
        frame_size, rot, trans, flags=cv.CALIB_ZERO_DISPARITY, alpha=rectify_scale)
    
    # Save calibration parameters
    np.save(os.path.join(output_dir, 'cameraMatrixL.npy'), cameraMatrixL)
    np.save(os.path.join(output_dir, 'cameraMatrixR.npy'), cameraMatrixR)
    np.save(os.path.join(output_dir, 'distL.npy'), distL)
    np.save(os.path.join(output_dir, 'distR.npy'), distR)
    np.save(os.path.join(output_dir, 'projMatrixL.npy'), projMatrixL)
    np.save(os.path.join(output_dir, 'projMatrixR.npy'), projMatrixR)
    np.save(os.path.join(output_dir, 'rot.npy'), rot)
    np.save(os.path.join(output_dir, 'trans.npy'), trans)
    np.save(os.path.join(output_dir, 'essentialMatrix.npy'), essentialMatrix)
    np.save(os.path.join(output_dir, 'fundamentalMatrix.npy'), fundamentalMatrix)
    np.save(os.path.join(output_dir, 'Q.npy'), Q)
    
    print("Calibration complete. Parameters saved in folder:", output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Stereo Calibration for the IMX219-83 Stereo Camera using Picamera2")
    parser.add_argument('--mode', type=str, choices=['capture', 'calibrate'],
                        required=True, help="Select 'capture' to acquire images or 'calibrate' to process calibration.")
    parser.add_argument('--num_images', type=int, default=20,
                        help="(capture mode) Number of stereo image pairs to capture")
    args = parser.parse_args()
    
    if args.mode == 'capture':
        capture_images(num_images=args.num_images)
    elif args.mode == 'calibrate':
        stereo_calibration()

if __name__ == '__main__':
    main()
