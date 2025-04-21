#!/usr/bin/env python3
"""
Calibrate intrinsic parameters for each camera using the images that
already exist in the `frames/` directory.  
For every camera key found in the calibration‑settings YAML (e.g. camera0,
camera1, …) it looks for files whose names start with that key inside
`frames/`, runs a chessboard calibration, and writes the intrinsics to
`camera_parameters/<camera>_intrinsics.dat`.

Usage:
    python3 calibrate_intrinsics_from_frames.py calibration_settings.yaml
"""
import cv2 as cv
import glob
import numpy as np
import sys
import yaml
import os

# Holds YAML data
calibration_settings = {}


def parse_calibration_settings_file(filename: str) -> None:
    """Load checkerboard settings (rows, cols, box size, …) from YAML."""
    global calibration_settings
    if not os.path.exists(filename):
        print("File does not exist:", filename)
        sys.exit(1)

    print("Using calibration settings file:", filename)
    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # Basic sanity check
    for key in ("checkerboard_rows", "checkerboard_columns", "checkerboard_box_size_scale"):
        if key not in calibration_settings:
            print(f'Missing key "{key}" in calibration settings')
            sys.exit(1)


def calibrate_camera_intrinsics(
    images_pattern: str, rows: int, columns: int, world_scaling: float
):
    """Run chessboard calibration for a set of images matching `images_pattern`."""
    image_files = glob.glob(images_pattern)
    if not image_files:
        print(f'No images found for pattern "{images_pattern}". Skipping.')
        return None, None

    images = [cv.imread(fname, cv.IMREAD_COLOR) for fname in image_files]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Prepare ideal object‑point grid
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp *= world_scaling

    objpoints, imgpoints = [], []

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ok, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ok:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

    if not objpoints:
        print(f'No valid chessboard detections for "{images_pattern}". Skipping.')
        return None, None

    h, w = images[0].shape[:2]
    rmse, cmtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    print(f"Calibration for pattern '{images_pattern}':")
    print("  RMSE:", rmse)
    print("  Camera matrix:\n", cmtx)
    print("  Distortion coefficients:", dist.ravel())
    return cmtx, dist


def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name: str) -> None:
    """Write intrinsics to camera_parameters/<camera>_intrinsics.dat."""
    if camera_matrix is None:
        return  # Nothing to save

    os.makedirs("camera_parameters", exist_ok=True)
    out_file = os.path.join("camera_parameters", f"{camera_name}_intrinsics.dat")

    with open(out_file, "w") as f:
        f.write("intrinsic:\n")
        for row in camera_matrix:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("distortion:\n")
        f.write(" ".join(map(str, distortion_coefs.ravel())) + "\n")

    print(f"Saved intrinsics to {out_file}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 calibrate_intrinsics_from_frames.py calibration_settings.yaml")
        sys.exit(1)

    # --- Load checkerboard settings ---
    parse_calibration_settings_file(sys.argv[1])

    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_columns"]
    scale = calibration_settings["checkerboard_box_size_scale"]

    # Any key that starts with “camera” is treated as a separate camera
    camera_names = [k for k in calibration_settings if k.startswith("camera")]
    if not camera_names:  # Fallback if YAML lacks explicit camera keys
        camera_names = ["camera0", "camera1"]

    # --- Calibrate each camera using pre‑saved frames ---
    for cam in camera_names:
        pattern = os.path.join("frames", f"{cam}*")
        cmat, dist = calibrate_camera_intrinsics(pattern, rows, cols, scale)
        save_camera_intrinsics(cmat, dist, cam)

    print("Intrinsic calibration complete.")


if __name__ == "__main__":
    main()
