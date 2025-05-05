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
```

Make sure you’ve also enabled the camera interfaces and installed the required PiCamera2 drivers from [here](https://github.com/raspberrypi/picamera2).

---

## 📋 Setup Instructions

1. **Print a checkerboard calibration pattern**: You can generate one [here](https://calib.io/pages/camera-calibration-pattern-generator).
2. **Update calibration_settings.yaml**:
   - Set the correct camera indices for `camera0` and `camera1`.
   - Adjust `frame_width`, `frame_height`, `checkerboard_rows`, `checkerboard_columns`, and `checkerboard_box_size_scale`.

---

## 🧪 Calibration Steps

### 1. Intrinsic Calibration (Per Camera)

Capture and calibrate intrinsic parameters:

```bash
python3 calibrate_intrinsics.py calibration_settings.yaml
```

This will:
- Capture frames from each camera.
- Detect checkerboard points.
- Save intrinsic matrices (`camera_parameters/camera*_intrinsics.dat`).

---

### 2. Stereo Calibration (Extrinsics)

Capture paired images and calibrate camera-to-camera transformation:

```bash
python3 calibrate_extrinsics.py calibration_settings.yaml
```

This will:
- Capture synchronized frames from both cameras.
- Compute and save rotation `R` and translation `T` between the cameras.

---

### 3. Visual Check

Verify the calibration by projecting 3D coordinate axes onto both camera views:

```bash
python3 check_calibration.py calibration_settings.yaml
```

---

## 📁 Folder Structure

```
.
├── calibration_settings.yaml
├── calibrate_intrinsics.py
├── calibrate_extrinsics.py
├── check_calibration.py
├── frames/
├── frames_pair/
└── camera_parameters/
```

---

## 🙏 Acknowledgements

This work builds directly on the excellent work by **Temuge B**, who originally created the stereo calibration framework and provided detailed documentation and explanations.

🔗 **Original GitHub Repo**: [https://github.com/TemugeB/python_stereo_camera_calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate)  
📝 **Blog Post**: [Stereo Calibration and Triangulation](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html)

We have modified the code to support **PiCamera2** on Raspberry Pi while preserving the calibration logic, structure, and design principles of the original project.

---

## 🐹 Bonus

This version sadly lacks a cute hamster co-developer like Milky from the original repo. Contributions welcome 😄

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file or visit [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0) for full terms.
