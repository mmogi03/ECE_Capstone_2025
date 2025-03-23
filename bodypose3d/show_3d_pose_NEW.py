import numpy as np
import matplotlib.pyplot as plt
from utils import DLT  # if needed elsewhere
plt.style.use('seaborn-v0_8')

# Use 68 facial landmarks from dlib.
num_keypoints = 68

def read_keypoints(filename):
    """Reads keypoints from file and reshapes each line into a 68 x N array."""
    fin = open(filename, 'r')
    kpts = []
    while True:
        line = fin.readline()
        if line == '':
            break
        line = line.split()
        line = [float(s) for s in line]
        line = np.reshape(line, (num_keypoints, -1))
        kpts.append(line)
    fin.close()
    kpts = np.array(kpts)
    return kpts

def visualize_3d(p3ds):
    """
    Visualizes 3D keypoints for the 68 dlib facial landmarks.
    The facial connectivity is defined as follows:
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
    right_eye = [[36,37], [37,38], [38,39], [39,40], [40,41], [41,36]]
    left_eye = [[42,43], [43,44], [44,45], [45,46], [46,47], [47,42]]
    outer_lip = [[48,49], [49,50], [50,51], [51,52], [52,53], [53,54],
                 [54,55], [55,56], [56,57], [57,58], [58,59], [59,48]]
    inner_lip = [[60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,60]]
    
    # Group all segments.
    face_segments = [jaw, right_eyebrow, left_eyebrow, nose_bridge, lower_nose,
                     right_eye, left_eye, outer_lip, inner_lip]
    # Assign a color for each segment.
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over each frame's 3D keypoints.
    for framenum, kpts3d in enumerate(p3ds):
        if framenum % 2 == 0:
            continue  # Skip every 2nd frame for smoother visualization
        for segment, seg_color in zip(face_segments, colors):
            for conn in segment:
                idx1, idx2 = conn
                ax.plot([kpts3d[idx1, 0], kpts3d[idx2, 0]],
                        [kpts3d[idx1, 1], kpts3d[idx2, 1]],
                        [kpts3d[idx1, 2], kpts3d[idx2, 2]],
                        linewidth=4, c=seg_color)

        # Optionally, you can add a scatter or text labels for each landmark:
        # for i in range(num_keypoints):
        #     ax.text(kpts3d[i, 0], kpts3d[i, 1], kpts3d[i, 2], str(i))
        #     ax.scatter(kpts3d[i, 0], kpts3d[i, 1], kpts3d[i, 2])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-10, 10)
        ax.set_xlabel('x')
        ax.set_ylim3d(-10, 10)
        ax.set_ylabel('y')
        ax.set_zlim3d(-10, 10)
        ax.set_zlabel('z')
        plt.pause(10.0)
        ax.cla()  # Clear the axes for the next frame

if __name__ == '__main__':
    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
