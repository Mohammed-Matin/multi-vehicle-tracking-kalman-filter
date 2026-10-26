# Multi-Vehicle Tracking Using Kalman Filter

A Python CLI utility that reads any input video, detects vehicles (YOLO), and tracks each
vehicle across frames using a per-vehicle Kalman Filter (Constant Velocity model).

- Unique track IDs per vehicle
- Handles occlusion via Kalman prediction
- Hungarian (IoU) data association
- Outputs an annotated video with bounding boxes, IDs, class, and position

Tech: Python, OpenCV, Ultralytics YOLO, SciPy.
