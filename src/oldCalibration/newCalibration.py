import cv2 as cv
from picamera2 import Picamera2
import yaml, os, time, numpy as np
import sys
from pathlib import Path
from typing import List


PATTERN_SIZE = (6, 8)

path = "./frames"


for cam in range (2):
    counter = 0
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    if cam == 0:
        subpath = '/camera0_frames/'
    else:
        subpath = '/camera1_frames/'
        
    wholepath = path + subpath

    listPaths = os.listdir(wholepath)

    for i in range(len(listPaths)):
        imgPath = wholepath + listPaths[i]
        img = cv.imread(imgPath)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, PATTERN_SIZE, None)
        #print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, PATTERN_SIZE, corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(500)
            counter +=1 
    cv.destroyAllWindows()
    print(len(imgpoints[0]))
    print(len(objpoints[0]))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    print("Camera Matrix: ")
    print(mtx)
    print(f"\nNumber of succesful corner finding images: {counter}.\nPrinting 3 succesful runs:")
    for j in range(3):
        print(f"\nRotation vectors {j+1}:")
        print(np.transpose(rvecs[j]))
        print(f"Translation vector {j+1}: ")
        print(np.transpose(tvecs[j]))
        
