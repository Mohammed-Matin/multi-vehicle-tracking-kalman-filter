import os
import cv2


def open_video(path):
    """Validate path and open a cv2.VideoCapture. Returns (cap, fps, w, h)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Input video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions: {w}x{h}")
    return cap, float(fps), w, h


def make_writer(path, fps, w, h):
    """Create a cv2.VideoWriter for the given output path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {path}")
    return writer