import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def undistort_image(img, camera_matrix, dist):
    height, width = img.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (width, height), 1, (width, height))
    undistorted = cv.undistort(img, camera_matrix, dist, None, new_camera_matrix)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]

def rotate_point_cloud(points, angle):
    # Rotate the point cloud about the Z-axis to align the eye-line horizontally.
    Rz = np.array([[np.cos(-angle), -np.sin(-angle), 0],
                   [np.sin(-angle),  np.cos(-angle), 0],
                   [0,               0,              1]])
    return points.dot(Rz.T)

def rotate_x(points, theta):
    # Rotate the point cloud about the X-axis.
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    return points.dot(Rx.T)

def main():
    # Load calibration parameters
    camera_matrix_L = np.load('../calibration/params/cameraMatrixL.npy')
    camera_matrix_R = np.load('../calibration/params/cameraMatrixR.npy')
    dist_L = np.load('../calibration/params/distL.npy')
    dist_R = np.load('../calibration/params/distR.npy')
    proj_matrix_L = np.load('../calibration/params/projMatrixL.npy')
    proj_matrix_R = np.load('../calibration/params/projMatrixR.npy')

    # Initialize face detector and predictor.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

    cap_left = cv.VideoCapture(1)
    cap_right = cv.VideoCapture(0, cv.CAP_DSHOW)

    # Create one matplotlib figure with three subplots.
    # ax1: left image, ax2: right image, ax3: 3D plot.
    fig = plt.figure("Combined Display")
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Left 2D")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Right 2D")
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title("3D Facial Landmarks")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    plt.ion()
    while True:
        retL, frameL = cap_left.read()
        retR, frameR = cap_right.read()
        if not retL or not retR:
            break

        undistortedL = undistort_image(frameL, camera_matrix_L, dist_L)
        undistortedR = undistort_image(frameR, camera_matrix_R, dist_R)

        # Convert BGR to RGB for matplotlib display.
        rgbL = cv.cvtColor(undistortedL, cv.COLOR_BGR2RGB)
        rgbR = cv.cvtColor(undistortedR, cv.COLOR_BGR2RGB)

        grayL = cv.cvtColor(undistortedL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(undistortedR, cv.COLOR_BGR2GRAY)

        landmarks_3d = []

        facesL = detector(grayL)
        facesR = detector(grayR)

        # Draw landmarks on left image.
        if facesL:
            shapeL = predictor(grayL, facesL[0])
            for i in range(68):
                ptL = (shapeL.part(i).x, shapeL.part(i).y)
                cv.circle(rgbL, ptL, 2, (0, 255, 0), -1)
        # Draw landmarks on right image.
        if facesR:
            shapeR = predictor(grayR, facesR[0])
            for i in range(68):
                ptR = (shapeR.part(i).x, shapeR.part(i).y)
                cv.circle(rgbR, ptR, 2, (0, 255, 0), -1)

        # Triangulate landmarks if faces are detected in both images.
        if facesL and facesR:
            shapeL = predictor(grayL, facesL[0])
            shapeR = predictor(grayR, facesR[0])
            for i in range(68):
                ptL = (shapeL.part(i).x, shapeL.part(i).y)
                ptR = (shapeR.part(i).x, shapeR.part(i).y)
                ptsL = np.array([[ptL[0]], [ptL[1]]], dtype=np.float64)
                ptsR = np.array([[ptR[0]], [ptR[1]]], dtype=np.float64)
                point4d = cv.triangulatePoints(proj_matrix_L, proj_matrix_R, ptsL, ptsR)
                point4d /= point4d[3]
                x, y, z = point4d[0][0], point4d[1][0], point4d[2][0]
                landmarks_3d.append((x, y, z))

        # Update subplot for left 2D image.
        ax1.cla()
        ax1.set_title("Left 2D")
        ax1.imshow(rgbL)
        ax1.axis("off")

        # Update subplot for right 2D image.
        ax2.cla()
        ax2.set_title("Right 2D")
        ax2.imshow(rgbR)
        ax2.axis("off")

        # Update the 3D plot.
        ax3.cla()
        ax3.set_title("3D Facial Landmarks")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        if landmarks_3d:
            points = np.array(landmarks_3d)
            # Center the points.
            mean = np.mean(points, axis=0)
            centered_points = points - mean

            # Compute average positions for the left eye (landmarks 36-41) and right eye (landmarks 42-47)
            left_eye = np.mean(centered_points[36:42], axis=0)
            right_eye = np.mean(centered_points[42:48], axis=0)
            eye_line = right_eye - left_eye
            # Compute rotation angle around Z (in XY plane)
            angle = np.arctan2(eye_line[1], eye_line[0])
            rotated_points = rotate_point_cloud(centered_points, angle)

            # Flip Z-axis so face originally pointing +Z comes toward the viewer
            rotated_points[:, 2] = -rotated_points[:, 2]

            # Rotate about the X-axis by +90 degrees to have the face point in -Y
            final_points = rotate_x(rotated_points, np.pi / 2)
            # Additional flip on Z if needed (here you can adjust as desired)
            final_points[:, 2] = -final_points[:, 2]

            # Dynamic plot limits
            max_range = np.max(np.abs(final_points)) * 1.2
            ax3.set_xlim([-max_range, max_range])
            ax3.set_ylim([-max_range, max_range])
            ax3.set_zlim([-max_range, max_range])

            xs, ys, zs = final_points[:, 0], final_points[:, 1], final_points[:, 2]
            ax3.scatter(xs, ys, zs, c='b', marker='o')
            # Mark the facial landmark with index 27 with a red triangle.
            landmark27 = final_points[27]
            ax3.scatter(landmark27[0], landmark27[1], landmark27[2], c='r', marker='^', s=100)
        else:
            ax3.set_xlim([-500, 500])
            ax3.set_ylim([-500, 500])
            ax3.set_zlim([-500, 500])
        plt.pause(0.001)
        # Optionally, break out on key press (this part requires additional handling if needed)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    plt.close()

if __name__ == "__main__":
    main()
