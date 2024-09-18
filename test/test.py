import cv2
from matplotlib import pyplot as plt
import numpy as np




cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
print("Complete")
cv2.destroyAllWindows()