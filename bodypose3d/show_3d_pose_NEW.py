import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# Use 68 facial landmarks from dlib.
num_keypoints = 68

def read_keypoints(filename):
    """Reads keypoints from file and reshapes each line into a (68 x N) array."""
    with open(filename, 'r') as fin:
        kpts = []
        for line in fin:
            line = line.split()
            line = [float(s) for s in line]
            line = np.reshape(line, (num_keypoints, -1))
            kpts.append(line)
    kpts = np.array(kpts)
    return kpts

def reorient_points(kpts3d):
    """
    Reorients the 3D points from camera coordinates to a more natural view.
    Transformation:
      new_x = original x
      new_y = original z   (depth becomes vertical)
      new_z = -original y  (flip the original vertical axis)
    """
    rotated = np.zeros_like(kpts3d)
    rotated[:, 0] = kpts3d[:, 0]       # x remains the same
    rotated[:, 1] = kpts3d[:, 2]       # z becomes new y
    rotated[:, 2] = -kpts3d[:, 1]      # -y becomes new z
    return rotated

def visualize_3d(p3ds, pause_time=0.1):
    """
    Visualizes 3D keypoints for the 68 dlib facial landmarks.
    Facial regions are connected as follows:
      - Jaw: points 0-16
      - Right eyebrow: points 17-21
      - Left eyebrow: points 22-26
      - Nose bridge: points 27-30
      - Lower nose: points 31-35
      - Right eye: points 36-41 (closed loop)
      - Left eye: points 42-47 (closed loop)
      - Outer lip: points 48-59 (closed loop)
      - Inner lip: points 60-67 (closed loop)
    """
    # Define connectivity for each facial region.
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
    
    # Group facial segments and assign colors.
    face_segments = [jaw, right_eyebrow, left_eyebrow, nose_bridge, lower_nose,
                     right_eye, left_eye, outer_lip, inner_lip]
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']

    from mpl_toolkits.mplot3d import Axes3D

    plt.ion()  # Enable interactive mode for smooth animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for framenum, kpts3d in enumerate(p3ds):
        # Reorient the 3D points for a natural view.
        rotated_kpts = reorient_points(kpts3d)

        # Plot each facial segment.
        for segment, seg_color in zip(face_segments, colors):
            for conn in segment:
                idx1, idx2 = conn
                # Skip if keypoints are invalid.
                if rotated_kpts[idx1, 0] == -1 or rotated_kpts[idx2, 0] == -1:
                    continue
                ax.plot([rotated_kpts[idx1, 0], rotated_kpts[idx2, 0]],
                        [rotated_kpts[idx1, 1], rotated_kpts[idx2, 1]],
                        [rotated_kpts[idx1, 2], rotated_kpts[idx2, 2]],
                        linewidth=2, c=seg_color)

        # Optionally, scatter the keypoints for clarity:
        # ax.scatter(rotated_kpts[:, 0], rotated_kpts[:, 1], rotated_kpts[:, 2])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-10, 10)
        ax.set_xlabel('x')
        ax.set_ylim3d(-10, 10)
        ax.set_ylabel('y')
        ax.set_zlim3d(-10, 10)
        ax.set_zlabel('z')
        
        plt.draw()
        plt.pause(pause_time)  # Short pause for smooth animation
        ax.cla()  # Clear axes for the next frame

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds, pause_time=0.1)
