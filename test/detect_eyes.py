import cv2
from matplotlib import pyplot as plt
import numpy as np
import serial
from time import sleep

i = 0
arduino = serial.Serial('COM4', 115200, timeout=0.5)
sleep(5)
angles = 180


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def move_servo():

    arduino.write("180".encode())
    sleep(2)




# gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)

# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 2)

# gray_face = gray_picture[y:y+h, x:x+w]
# face = img[y:y+h, x:x+w]
# eyes = eye_cascade.detectMultiScale(gray_face)

# for (ex, ey, ew, eh) in eyes:
#     cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (0,255,255),2)

def detect_eyes(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5) # detect eyes
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye  

def detect_faces(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]

    return img

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    calibration_params = np.load("../calibration/calibration_params/calibration.npz")
    ret, mtx, dist, rvecs, tvecs = calibration_params["ret"], calibration_params["mtx"], calibration_params["dist"], calibration_params["rvecs"], calibration_params["tvecs"]

    print(f"Camera matrix:\n{mtx}")

    # NOTE: try to figure out which one is right vs left eye, then find midpoint? it's not well tuned and dynamic, so keep storing last position
    last_position_left_eye = None
    last_position_right_eye = None
    last_position_midpt_eye = None


    while True:

        _, frame = cap.read()

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort -- last argument COULD be changed to newcameramtx
        # frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # crop image
        # x, y, w, h = roi
        # frame = frame[y:y+h, x:x+w]

        face_frame = detect_faces(frame, face_cascade)

        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            
            for i, eye in enumerate(eyes):
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)

                    if len(keypoints) > 0 and i == 0:
                        last_position_left_eye = keypoints[0].pt
                        print("left: ", keypoints[0].pt)
                        move_servo()



                    elif len(keypoints) > 0 and i == 1:
                        last_position_right_eye = keypoints[0].pt
                        print("right: ", keypoints[0].pt)
                        move_servo()



                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow('image', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if last_position_left_eye and last_position_right_eye:
            last_position_midpt_eye = (int((last_position_left_eye[0] + last_position_right_eye[0]) / 2), int((last_position_left_eye[1] + last_position_right_eye[1]) / 2))

            # get the ray projecting into pixel in 3D space
            Ki = np.linalg.inv(mtx)
            r = Ki.dot([last_position_midpt_eye[0], last_position_midpt_eye[1], 1.0])
            print("r = ", r)

            # cv2.circle(frame, last_position_midpt_eye, radius=2, color=(0, 0, 255), thickness=-1)
    
    cap.release()
    print("Complete")
    cv2.destroyAllWindows()

def nothing(x):
    pass

# need to adjust threshold... maybe dynamically?
# _, img = cv2.threshold(img, 88, 255, cv2.THRESH_BINARY)

# cv2.imshow("output img", img)

# face = detect_faces(img, None)
# cv2.imshow("face", face)


main()
arduino.close()
