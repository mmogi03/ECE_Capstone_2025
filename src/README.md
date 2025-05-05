# DLT Triangulation Utility (from BodyPose3D)

This file is a utility module adapted from the original [`utils.py`](https://github.com/TemugeB/bodypose3d/blob/main/utils.py) in the [**BodyPose3D**](https://github.com/TemugeB/bodypose3d) repository by **Temuge Batpurev**.

---

## 🙋‍♂️ What This Is

We are using the **Direct Linear Transform (DLT)** method from the original `utils.py` to triangulate 3D points using stereo camera projection matrices. Specifically, we use:

- `DLT(P1, P2, point1, point2)`  
  → Triangulates a 3D point from 2D correspondences and projection matrices.

- `_make_homogeneous_rep_matrix(R, t)`  
  → Constructs a homogeneous transformation matrix from rotation and translation.

We do **not** use other functionality from the original `utils.py` file (e.g., reading calibration files or writing keypoints to disk).

---

## 📌 Original Project

- 📂 **Repo**: [https://github.com/TemugeB/bodypose3d](https://github.com/TemugeB/bodypose3d)
- ✍️ **Author**: [Temuge Batpurev](https://github.com/TemugeB)
- 📄 **Source File Used**: [`utils.py`](https://github.com/TemugeB/bodypose3d/blob/main/utils.py)

Temuge’s BodyPose3D is an open-source multi-camera 3D human pose estimation system. We acknowledge and thank the author for open-sourcing this work.

---

## 📁 Originality of the Rest of the Code

All other code in the [`src/`](../src/) folder—including the various inference, server, serial communication, and calibration scripts—is **entirely original** to this project.

For an overview of the full system and setup instructions, please refer to the [**project-level README**](../README.md).

---

## 📜 License

This project incorporates code under the [MIT License](LICENSE), as permitted by the original BodyPose3D repository.
