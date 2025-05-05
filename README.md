# Attention-Aware Adjustable Car Mirror System

This repository contains the complete codebase and resources for a senior capstone project aimed at creating an **autonomously adjustable rearview mirror** system that enhances automobile safety by aligning the mirror based on the driver's eye position.

## 🚗 Project Overview

This project integrates computer vision, deep learning, embedded systems, and mechatronic actuation to develop an intelligent rearview mirror that dynamically adjusts based on the driver’s eye position. The system utilizes **stereo vision cameras**, **facial landmark detection**, and **3D triangulation** to determine the driver’s relative position and computes **pitch and yaw angles** to orient the mirror accordingly.

The core pipeline runs on a **Raspberry Pi 5**, while the **mirror actuation is controlled via Arduino**. A companion **mobile app** (hosted as a Git submodule) offers manual override and real-time control, ensuring driver flexibility and personalization.

For full technical details, read our [final project report (Capstone_Final_Report.pdf)](./Capstone_Final_Report.pdf).

---

## 🧠 Technologies Used

- **Stereo Vision** (via Picamera2)
- **Facial Landmark Detection** (Dlib 68-point model)
- **3D Triangulation** (DLT algorithm)
- **Embedded Systems** (Arduino, Serial Communication)
- **PID Motor Control**
- **WebSockets** for real-time client-server communication
- **Mobile App (React Native)** for manual control & presets
- **Raspberry Pi 5** as the backend compute node

### 🧰 Libraries & Frameworks

From the codebase, the following key Python libraries and frameworks are used:

- `opencv-python (cv2)` – image processing and visualization  
- `numpy` – vector and matrix operations  
- `picamera2` – Raspberry Pi stereo camera capture  
- `dlib` – facial landmark detection  
- `sympy` – symbolic geometry for mirror angle calculations  
- `scipy.linalg` – SVD used in triangulation  
- `json` – data communication with the Arduino  
- `asyncio`, `websockets` – asynchronous server/client communication  
- `multiprocessing` – parallel inference and control  
- `serial` – serial interface to Arduino  
- `ArduinoJson` (on Arduino side) – JSON parsing for embedded C++

Ensure all dependencies are installed via `requirements.txt` or your preferred package manager.

---

## 🗂 Repository Structure

```
/
├── arduino_code/           # Arduino codebase for motor control
├── camera_calibration/     # Calibration scripts + README and LICENSE
├── src/                    # Main backend & inference system (hosted on Pi)
├── src_old/                # [DEPRECATED] Previous implementation (ignore)
├── Capstone_MirrorApp/     # Mobile App (Git submodule to external repo)
```

### 🔧 Setup Instructions (Main Backend Code)

Please follow the instructions under [`src/`](./src) for full details on backend setup, including calibration steps and runtime.

To **start** the real-time server and inference pipeline, run:

```bash
./src/start_service_all.sh
```

To **kill** all processes (including lingering PID processes), run:

```bash
./src/kill_stuff.sh
```

For full setup, calibration, and configuration instructions — go to the [src/ directory README](./src/README.md).

---

## 🤖 Arduino Motor Controller

Motor actuation is controlled by an Arduino running a dual-axis **PID loop**, with pitch and yaw controlled using encoder feedback. The firmware ensures the mirror rotates smoothly to desired angles, either automatically from deep-learning inference or manually via the app.

The main script for the mirror controller is in:
```
arduino_code/Arduino_final/pid_final_v1.ino
```

It handles:

- Encoder feedback
- Dual-axis PID (tunable)
- Serial communication with Raspberry Pi
- Motor driver control

---

## 📱 Capstone_MirrorApp — Mobile Client

The folder [`Capstone_MirrorApp`](./Capstone_MirrorApp) is a **Git submodule** that links to our external repository hosting the **React Native mobile app**.

This mobile interface connects to the Raspberry Pi backend via **WebSockets**, allowing:

- Manual control of mirror pose
- Save/load presets per user
- Real-time updates
- Firebase Authentication

> Note: Ensure submodules are initialized via `git submodule update --init` after cloning this repository.

---

## 📸 Camera Calibration

Our stereo calibration utilities reside under:

```
camera_calibration/
```

These scripts support intrinsic and extrinsic calibration using OpenCV with an 8×6 checkerboard. A `requirements.txt` file is included for dependencies. The deep-learning pipeline depends on these results to compute accurate triangulations.

---

## 🤝 Acknowledgements

We acknowledge the use of the [BodyPose3D](https://github.com/TemugeB/bodypose3d) project under the [MIT License](https://github.com/TemugeB/bodypose3d/blob/main/LICENSE), specifically for the **DLT triangulation function** from `utils.py`. This function was adapted in our backend 3D landmark reconstruction logic.

All other code in the `src/` folder is **original and developed by our team**. For a breakdown of each script and its usage, see the [src-level README](./src/README.md).

---

## 📄 Licensing

- All original work in this repository is licensed under the **Apache 2.0 License**. See the [LICENSE](./LICENSE) file for terms.
- The DLT logic from BodyPose3D is used under the **MIT License**.

---

## 📌 Contributors

- Cristian Llerena  
- Michael Mogilevsky  
- Adarsh Narayanan  
- Muralii Krishnan Thirumalai  
- Hazem Zaky  

Advisors:  
- Dr. Kristin Dana  
- Dr. Maria Striki  
- Dr. Zhao Zhang

---

## 📚 Citation

If you use this codebase or find it helpful in your research or products, please cite the project or link back to this repository.

---

**© 2025 Rutgers ECE Capstone Team — All rights reserved**
