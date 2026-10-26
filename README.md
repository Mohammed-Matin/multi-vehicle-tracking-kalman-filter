# Multi-Vehicle Tracking Using Kalman Filter

A Python CLI utility that reads any input video, detects vehicles (YOLOv8), and tracks each
vehicle across frames using a per-vehicle Kalman Filter (Constant Velocity model).

## Features
- Unique track ID per vehicle
- Kalman predict + update (Constant Velocity)
- Mahalanobis-distance data association (Hungarian algorithm)
- N-frame confirmation gate (filters one-frame false positives)
- Occlusion handling: predicts position when a vehicle is missing, ages out lost tracks
- Per-track color, trajectory trail (optional), predicted box shown during occlusion
- Annotated output video saved to disk

## Setup
```bash
pip install -r requirements.txt
```
YOLOv8n weights (`yolov8n.pt`) download automatically on first run.

## Usage
```bash
python vehicle_tracker.py --input path/to/video.mp4 --output out.mp4
```
Options:
- `--conf 0.4` YOLO confidence threshold
- `--trail 15` enable trajectory trail (default off)
- `--no-show` disable live preview (useful on headless/WSL)

## Test videos
Stored locally in `D:\WSL\Videos\AIR` (not committed). Sources: intel-iot-devkit sample-videos + YouTube traffic clips.
`download_videos.sh` documents how the YouTube clips were fetched (requires yt-dlp + deno).

## References
- MATLAB: Motion-Based Multiple Object Tracking (Computer Vision Toolbox)
- SORT / DeepSORT association ideas