import numpy as np
import cv2 as cv
import glob
from sympy import Plane, Point3D
import math
import serial
from time import sleep
import dlib  # UPDATED: using dlib for facial landmark detection

def generate_calibration_params():
    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
    chessboardSize = (8,6)
    frameSize = (640,480)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 30
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in image plane.
    imgpointsR = []  # 2d points in image plane.

    imagesLeft = sorted(glob.glob('../calibration/images/stereoLeft/*.png'))
    imagesRight = sorted(glob.glob('../calibration/images/stereoright/*.png'))

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:
            objpoints.append(objp)

            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv.imshow('img left', imgL)
            cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv.imshow('img right', imgR)
            cv.waitKey(1000)

    cv.destroyAllWindows()

    ############## CALIBRATION #######################################################
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    ########## Stereo Vision Calibration #############################################
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camera matrices so that only Rot, Trans, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same 

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # This step is performed to determine transformation between the two cameras and calculate Essential and Fundamental matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    ########## Stereo Rectification #################################################
    rectifyScale = 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
        newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0,0))

    with open('../calibration/params/distL.npy', 'wb') as f:
        np.save(f, distL)
    with open('../calibration/params/distR.npy', 'wb') as f:
        np.save(f, distR)
    with open('../calibration/params/projMatrixL.npy', 'wb') as f:
        np.save(f, projMatrixL)
    with open('../calibration/params/projMatrixR.npy', 'wb') as f:
        np.save(f, projMatrixR)
    with open('../calibration/params/cameraMatrixL.npy', 'wb') as f:
        np.save(f, cameraMatrixL)
    with open('../calibration/params/cameraMatrixR.npy', 'wb') as f:
        np.save(f, cameraMatrixR)

def undistort_frame(img1, img2):
    height, width = img1.shape[:2]

    camera_matrix_L = np.load('../calibration/params/cameraMatrixL.npy')
    camera_matrix_R = np.load('../calibration/params/cameraMatrixR.npy')
    dist_L = np.load('../calibration/params/distL.npy')
    dist_R = np.load('../calibration/params/distR.npy')
    proj_matrix_L = np.load('../calibration/params/projMatrixL.npy')
    proj_matrix_R = np.load('../calibration/params/projMatrixR.npy')

    # get optimal camera matrix for better undistortion
    new_camera_matrix_L, roi_L = cv.getOptimalNewCameraMatrix(camera_matrix_L, dist_L, (width, height), 1, (width, height))
    new_camera_matrix_R, roi_R = cv.getOptimalNewCameraMatrix(camera_matrix_R, dist_R, (width, height), 1, (width, height))

    new_img1 = cv.undistort(img1, camera_matrix_L, dist_L, None, new_camera_matrix_L)
    new_img2 = cv.undistort(img2, camera_matrix_R, dist_R, None, new_camera_matrix_R)

    roi_L_x, roi_L_y, roi_L_width, roi_L_height = roi_L
    new_img1 = new_img1[roi_L_y:roi_L_y + roi_L_height, roi_L_x:roi_L_x + roi_L_width]

    roi_R_x, roi_R_y, roi_R_width, roi_R_height = roi_R
    new_img2 = new_img2[roi_R_y:roi_R_y + roi_R_height, roi_R_x:roi_R_x + roi_R_width]

    return new_img1, new_img2

def main():
    # UPDATED: Load dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
    
    # (Cascade classifiers remain in the file below but are no longer used)
    face_cascade = cv.CascadeClassifier('../CascadeClassifiers/haarcascade_frontalface_alt.xml')
    eye_cascade = cv.CascadeClassifier('../CascadeClassifiers/haarcascade_eye_tree_eyeglasses.xml')
    
    proj_matrix_L = np.load('../calibration/params/projMatrixL.npy')
    proj_matrix_R = np.load('../calibration/params/projMatrixR.npy')

    vid_cap_L = cv.VideoCapture(0)
    vid_cap_R = cv.VideoCapture(1, cv.CAP_DSHOW)

    while True:
        ret_L, img_L = vid_cap_L.read()
        ret_R, img_R = vid_cap_R.read()
        
        # Undistort the frames
        undistorted_L, undistorted_R = undistort_frame(img_L, img_R)
        gray_L = cv.cvtColor(undistorted_L, cv.COLOR_BGR2GRAY)
        gray_R = cv.cvtColor(undistorted_R, cv.COLOR_BGR2GRAY)
        
        # UPDATED: Use dlib to detect faces and extract facial landmarks
        faces_L = detector(gray_L)
        faces_R = detector(gray_R)
        
        if len(faces_L) > 0 and len(faces_R) > 0:
            # Use the first detected face in each frame
            shape_L = predictor(gray_L, faces_L[0])
            shape_R = predictor(gray_R, faces_R[0])
            
            # Get the 28th point from the landmark array (assuming 1-indexed, so index 27 in Python)
            midpt_L = (shape_L.part(27).x, shape_L.part(27).y)
            midpt_R = (shape_R.part(27).x, shape_R.part(27).y)
            
            # Draw the landmark point on the images for visualization
            cv.circle(img_L, midpt_L, radius=2, color=(0,255,0), thickness=2)
            cv.circle(img_R, midpt_R, radius=2, color=(0,255,0), thickness=2)
            
            print("midpt_L: ", midpt_L)
            print("midpt_R: ", midpt_R)
            
            # Triangulate the 3D point using the projection matrices
            threeD_midpt = cv.triangulatePoints(proj_matrix_L, proj_matrix_R, midpt_L, midpt_R)
            threeD_midpt /= threeD_midpt[3]
            print("3D midpoint: ", threeD_midpt, "\n")
            
            try:
                # Manually set the rear window point -- NEED TO MANUALLY ADJUST TO MATCH RELATIVE TO MIRROR ITSELF
                threeD_window_pt = [[ -5e+05],
                                    [ 8.65304694e+04],
                                    [-1.1e+04],
                                    [ 1.00000000e+00]]

                driver_plane = Plane(Point3D(0, 0, 0), 
                                     Point3D(threeD_midpt[0][0], 0, threeD_midpt[2][0]), 
                                     Point3D(threeD_midpt[0][0], threeD_midpt[1][0], threeD_midpt[2][0]))
                rear_window_plane = Plane(Point3D(0, 0, 0), 
                                          Point3D(threeD_window_pt[0][0], 0, threeD_window_pt[2][0]), 
                                          Point3D(threeD_window_pt[0][0], threeD_window_pt[1][0], threeD_window_pt[2][0]))
                print("Line 266\n")
                xy_plane = Plane(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
                xz_plane = Plane(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))

                alpha = driver_plane.angle_between(rear_window_plane)
                beta = driver_plane.angle_between(xy_plane)
                yaw = math.degrees(alpha/2 + beta)
                print("Line 273 \n")
                bisector_pt1 = tuple(driver_plane.intersection(rear_window_plane)[0].points[0])
                bisector_pt2 = tuple(driver_plane.intersection(rear_window_plane)[0].points[1])
                bisector_pt3 = (1, 0, math.tan(yaw))
                print("We get to line 277")
                bisector_plane = Plane(Point3D(bisector_pt1[0], bisector_pt1[1], bisector_pt1[2]), 
                                       Point3D(bisector_pt2[0], bisector_pt2[1], bisector_pt2[2]), 
                                       Point3D(bisector_pt3[0], bisector_pt3[1], bisector_pt3[2]))
                
                driver_mirror_plane = Plane(Point3D(0, 0, 0), 
                                            Point3D(threeD_midpt[0][0], threeD_midpt[1][0], threeD_midpt[2][0]), 
                                            Point3D(threeD_window_pt[0][0], threeD_window_pt[1][0], threeD_window_pt[2][0]))

                final_line = bisector_plane.intersection(driver_mirror_plane)
                pitch = math.degrees(driver_mirror_plane.angle_between(xz_plane))
                print("Line 288\n")
                print(pitch, " ", yaw, "\n")
                ser.write(f"{pitch}".encode())
                sleep(2)
                ser.write(f"{yaw}".encode())
                sleep(2)

                print("Completed writing serial. Terminating...")
                return 0
            except Exception as error:
                print("Michael's math is wrong", error)
                break
        else:
            print("No face detected in one or both cameras.")
        
        cv.imshow('Left Camera', img_L)
        cv.imshow('Right Camera', img_R)
        if cv.waitKey(2) & 0xFF == ord('q'):
            break

#generate_calibration_params()
ser = serial.Serial("COM4", 115200, timeout=0.5)
main()