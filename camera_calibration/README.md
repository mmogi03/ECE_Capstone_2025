# Stereo Camera Calibration with PiCamera2 Support

This project is a modified version of [Temuge B's stereo camera calibration repository](https://github.com/TemugeB/python_stereo_camera_calibrate), originally written for USB cameras on Linux using OpenCV. This forked version has been adapted specifically for use with Raspberry Pi systems utilizing **PiCamera2**.

> **Original Author:** [Temuge B](https://github.com/TemugeB)  
> **Original Repo:** https://github.com/TemugeB/python_stereo_camera_calibrate  
> **Original Blog Post:** [Stereo Camera Calibration and Triangulation](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html)

---

## 🚀 What’s New / Modified

This fork includes the following key enhancements:

- ✅ **PiCamera2 Integration**: All camera operations now use the [Picamera2](https://github.com/raspberrypi/picamera2) Python library.
- ✅ **Modular Workflow**: Split the calibration into three distinct stages:
  - `calibrate_intrinsics.py` — Intrinsic calibration (per camera)
  - `calibrate_extrinsics.py` — Stereo extrinsic calibration
  - `check_calibration.py` — Final visual validation of calibration result
- ✅ **Cross-Platform Paths & Clean Output**: Improved file management, logging, and YAML parsing.
- ✅ **Index Resumption**: Paired image capture resumes from the last index to avoid overwriting.
- ✅ **More Robust Error Handling**: Added checks and messages to prevent silent failures.
- ✅ **Improved Documentation & Comments**: For easier readability and maintenance.

---

## 🧰 Requirements

- Raspberry Pi (any model with camera ports)
- PiCamera2 library (`libcamera` backend)
- Python 3.8+
- Packages: `opencv-python`, `pyyaml`, `numpy`, `scipy`, `picamera2`

Install all dependencies:

```bash
pip3 install -r requirements.txt
