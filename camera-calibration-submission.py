#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified on 2024-03-20 from tutorial code created by paavanasrinivas
"""

import numpy as np
import cv2 as cv
import glob


################ DEFINE CHECKERBOARD CHARACTERISTICS #############################

# Defining the dimensions of the checkerboard

"""
Adjust these based on your image characteristics!

"""

chessboardSize = (7, 10) # Use (Rows-1) & (Columns-1)
frameSize = (1920, 1080) # Dimensions of your image set
size_of_chessboard_squares_mm = 25 # Length of one square

################ FIND CHESSBOARD CORNERS - OBJECT AND IMAGE POINTS #############################

# Defining the termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Preparing object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * size_of_chessboard_squares_mm


# Creating a vector to store vectors of 3D points for each checkerboard image in real-world space
objpoints = [] 

# Creating a vector to store vectors of 2D points in image place for each checkerboard image 
imgpoints = [] 

# Extracting the path of individual images stored in a given directory
images = glob.glob('./images/*.jpg') 
""" INSERT THE PATH TO YOUR IMAGE DIRECTORY ABOVE!! """

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If desired number of corners are detected, refine the pixel coordinates and display them on the images of the checkerboard
    if ret == True:

        objpoints.append(objp)
        # Refining pixel coordinates for given 2d points.
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

        imS = cv.resize(img, (800, 450)) # Resize image
        
        cv.imshow('img', imS)
        cv.waitKey(2000)


cv.destroyAllWindows()



############## CAMERA CALIBRATION #######################################################

# Performing camera calibration by passing the value of known 3D points (objpoints) and corresponding pixel coordinates of the detected corners (imgpoints)
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Intrinsic Parameters
print("Camera matrix : \n")
print(cameraMatrix)
print("Distance : \n")
print(dist)

# Extrinsic Parameters
print("Rotation Vectors (rvecs): \n")
print(rvecs)
print("tTranslation Vectors (vecs): \n")
print(tvecs)


############## UNDISTORTION #####################################################

"""
Insert code to Undistort one image from your set and save it.
Hint:
        1. Read an image from your directory
        2. Use cv.getOptimalNewCameraMatrix (or) cv.initUndistortRectifyMap
        3. Crop the image based on the region of interest (roi)
        4. Save the result
        
"""

# read an image
img = cv.imread("./images/WIN_20240318_23_38_45_Pro.jpg")
h, w = img.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# save the result
cv.imwrite('calibresult.png', dst)

############## CALCULATE RE-PROJECTION ERROR #####################################################

# Re-projection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

total_error = mean_error / len(objpoints)
print(f"Total Re-projection Error: {total_error}")
