#!/usr/bin/env python3
"""
Capture 20 frames from each eye of an IMX219‑83 stereo module and save them in
separate folders:

    frames/
        camera0_frames/ img_01.png … img_20.png
        camera1_frames/ img_01.png … img_20.png

Press <SPACE> to start capturing; press <ESC> to abort the program.
"""

import cv2 as cv
from picamera2 import Picamera2
import yaml, os, time

# ─── SETTINGS ──────────────────────────────────────────────────────────────────
SETTINGS_FILE = "./calibration_settings.yaml"   # still used to fetch camera indices
ROOT_DIR      = "frames"                      # root folder for all images
N_IMAGES      = 10                            # images to store per eye
# ───────────────────────────────────────────────────────────────────────────────

def load_settings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_settings(SETTINGS_FILE)

# Basic frame parameters (provide sensible fallbacks)
WIDTH         = cfg.get("frame_width",   640)
HEIGHT        = cfg.get("frame_height",  480)
VIEW_RESIZE   = cfg.get("view_resize",   1)
COOLDOWN_TIME = cfg.get("cooldown",     30)   # frames to wait between saves

os.makedirs(ROOT_DIR, exist_ok=True)

for cam_id in range(2):                                      # camera0, camera1
    cam_key   = f"camera{cam_id}"
    cam_idx   = cfg.get(cam_key, cam_id)                     # device index "/dev/videoX"
    save_dir  = os.path.join(ROOT_DIR, f"{cam_key}_frames")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== {cam_key}  (device index {cam_idx}) ===")
    picam2 = Picamera2(camera_num=cam_idx)
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT)}))
    picam2.start()

    start        = False
    cooldown     = COOLDOWN_TIME
    saved_count  = 0

    while True:
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            print("No data from camera. Exiting…")
            break

        frame_bgr   = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        frame_small = cv.resize(frame_bgr, None,
                                fx=1/VIEW_RESIZE, fy=1/VIEW_RESIZE)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start capture",
                       (50, 50), cv.FONT_HERSHEY_COMPLEX,
                       1, (0, 0, 255), 1)
        else:
            cooldown -= 1
            cv.putText(frame_small, f"Cooldown: {cooldown}",
                       (50, 50),  cv.FONT_HERSHEY_COMPLEX,
                       1, (0, 255, 0), 1)
            cv.putText(frame_small, f"Captured: {saved_count}/{N_IMAGES}",
                       (50, 100), cv.FONT_HERSHEY_COMPLEX,
                       1, (0, 255, 0), 1)

            if cooldown <= 0:
                filename = os.path.join(save_dir,
                                        f"img_{saved_count+1:02d}.png")
                cv.imwrite(filename, frame_bgr)
                print(f"Saved {filename}")
                saved_count += 1
                cooldown = COOLDOWN_TIME

        cv.imshow(f"{cam_key} preview", frame_small)

        key = cv.waitKey(1) & 0xFF
        if key == 27:                 # ESC → quit completely
            picam2.stop(); cv.destroyAllWindows(); quit()
        elif key == 32:               # SPACE → begin capture loop
            start = True

        if saved_count >= N_IMAGES:   # done with this eye
            break

    picam2.stop()
    cv.destroyAllWindows()
    time.sleep(2)                     # short pause before switching eyes

print("✔ Captured 20 images for each camera; files saved in 'frames/'")



