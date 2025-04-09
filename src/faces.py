import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt

import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from utils import DLT, get_projection_matrix

from PIL import Image, ImageTk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Desired frame resolution: height x width
frame_shape = [1232, 1640]  # (height=1232, width=1640)

def reorient_points(kpts3d):
    """
    Reorients the 3D points for a natural view:
      new_x = original x
      new_y = original z   (depth becomes vertical)
      new_z = - original y (flip the original vertical axis)
    """
    rotated = np.zeros_like(kpts3d)
    rotated[:, 0] = kpts3d[:, 0]       # x remains the same
    rotated[:, 1] = kpts3d[:, 2]         # original z becomes new y
    rotated[:, 2] = -kpts3d[:, 1]        # -original y becomes new z
    return rotated

def main():
    # Initialize the two cameras using Picamera2.
    camera_index0 = 0
    camera_index1 = 1

    picam2_0 = Picamera2(camera_num=camera_index0)
    config0 = picam2_0.create_preview_configuration(main={"size": (frame_shape[1], frame_shape[0])})
    picam2_0.configure(config0)
    picam2_0.start()

    picam2_1 = Picamera2(camera_num=camera_index1)
    config1 = picam2_1.create_preview_configuration(main={"size": (frame_shape[1], frame_shape[0])})
    picam2_1.configure(config1)
    picam2_1.start()

    # Initialize dlib face detector and the 68 facial landmark predictor.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Get projection matrices (ensure these are correctly calibrated).
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    # Create the main Tkinter window.
    root = tk.Tk()
    root.title("Combined Camera Views and 3D Plot")

    # Create frames for camera views and the 3D plot.
    cam_frame = tk.Frame(root)
    cam_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Define desired display sizes for the camera feeds.
    cam_width = 320
    cam_height = 240

    # Create two labels for the two camera frames.
    label_cam0 = tk.Label(cam_frame)
    label_cam0.pack(side=tk.LEFT, padx=10, pady=10)
    label_cam1 = tk.Label(cam_frame)
    label_cam1.pack(side=tk.LEFT, padx=10, pady=10)

    # Set up the matplotlib figure for 3D plotting and embed it in Tkinter.
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    canvas_plot = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Define facial segment connectivity.
    jaw = [[i, i+1] for i in range(0, 16)]
    right_eyebrow = [[i, i+1] for i in range(17, 21)]
    left_eyebrow = [[i, i+1] for i in range(22, 26)]
    nose_bridge = [[i, i+1] for i in range(27, 30)]
    lower_nose = [[i, i+1] for i in range(31, 35)]
    right_eye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36]]
    left_eye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42]]
    outer_lip = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
                 [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48]]
    inner_lip = [[60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]]
    face_segments = [jaw, right_eyebrow, left_eyebrow, nose_bridge, lower_nose,
                     right_eye, left_eye, outer_lip, inner_lip]
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']

    def update():
        # Capture frames from each camera.
        frame0 = picam2_0.capture_array()
        frame1 = picam2_1.capture_array()

        if frame0 is None or frame1 is None:
            root.after(10, update)
            return

        # Optionally crop to a square (if needed).
        if frame0.shape[1] != frame_shape[0]:
            start_col = frame_shape[1] // 2 - frame_shape[0] // 2
            end_col = frame_shape[1] // 2 + frame_shape[0] // 2
            frame0 = frame0[:, start_col:end_col]
            frame1 = frame1[:, start_col:end_col]

        # Convert to grayscale for face detection.
        gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        # Detect faces in both frames.
        rects0 = detector(gray0, 1)
        rects1 = detector(gray1, 1)

        # Process camera 0 landmarks.
        frame0_keypoints = []
        if len(rects0) > 0:
            shape0 = predictor(gray0, rects0[0])
            for i in range(68):
                x, y = shape0.part(i).x, shape0.part(i).y
                frame0_keypoints.append([x, y])
                cv.circle(frame0, (x, y), 2, (0, 0, 255), -1)
        else:
            frame0_keypoints = [[-1, -1]] * 68

        # Process camera 1 landmarks.
        frame1_keypoints = []
        if len(rects1) > 0:
            shape1 = predictor(gray1, rects1[0])
            for i in range(68):
                x, y = shape1.part(i).x, shape1.part(i).y
                frame1_keypoints.append([x, y])
                cv.circle(frame1, (x, y), 2, (0, 0, 255), -1)
        else:
            frame1_keypoints = [[-1, -1]] * 68

        # Compute 3D keypoints using DLT for each landmark pair.
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                p3d = [-1, -1, -1]
            else:
                p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(p3d)
        frame_p3ds = np.array(frame_p3ds)  # Shape: (68, 3)

        # Update the 3D plot.
        ax.cla()  # Clear previous frame.
        rotated_kpts = reorient_points(frame_p3ds)
        for segment, seg_color in zip(face_segments, colors):
            for conn in segment:
                idx1, idx2 = conn
                # Skip if keypoints are not valid.
                if rotated_kpts[idx1, 0] == -1 or rotated_kpts[idx2, 0] == -1:
                    continue
                ax.plot([rotated_kpts[idx1, 0], rotated_kpts[idx2, 0]],
                        [rotated_kpts[idx1, 1], rotated_kpts[idx2, 1]],
                        [rotated_kpts[idx1, 2], rotated_kpts[idx2, 2]],
                        linewidth=2, c=seg_color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(-10, 10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        canvas_plot.draw()

        # Resize the camera frames to the desired display size.
        frame0_resized = cv.resize(frame0, (cam_width, cam_height))
        frame1_resized = cv.resize(frame1, (cam_width, cam_height))

        # Update the camera frame images in the Tkinter labels.
        frame0_rgb = cv.cvtColor(frame0_resized, cv.COLOR_BGR2RGB)
        frame1_rgb = cv.cvtColor(frame1_resized, cv.COLOR_BGR2RGB)
        img0 = Image.fromarray(frame0_rgb)
        img1 = Image.fromarray(frame1_rgb)
        imgtk0 = ImageTk.PhotoImage(image=img0)
        imgtk1 = ImageTk.PhotoImage(image=img1)
        label_cam0.imgtk = imgtk0  # Keep a reference.
        label_cam1.imgtk = imgtk1
        label_cam0.configure(image=imgtk0)
        label_cam1.configure(image=imgtk1)

        # Schedule the next update.
        root.after(10, update)

    update()  # Start the update loop.
    root.mainloop()

    # Cleanup.
    picam2_0.stop()
    picam2_1.stop()
    plt.close(fig)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
