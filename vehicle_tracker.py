import argparse
import os

import cv2
import numpy as np

from utils.video_io import open_video, make_writer
from tracker.detector import VehicleDetector
from tracker.association import MultiVehicleTracker


def color_for_id(tid):
    rng = np.random.default_rng(tid * 9973)
    return tuple(int(x) for x in rng.integers(60, 230, size=3))


def draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash=8):
    x1, y1 = pt1
    x2, y2 = pt2
    # top & bottom
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    # left & right
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def run_pipeline(input_path, output_path, conf=0.4, trail=0, show=True):
    cap, fps, w, h = open_video(input_path)
    writer = make_writer(output_path, fps, w, h)

    detector = VehicleDetector(conf_thresh=conf)
    tracker = MultiVehicleTracker(max_age=20, mahal_thresh=9.49,
                                 confirm_frames=3, tentative_age=5)

    frame_count = 0
    print(f"Processing {input_path} ({w}x{h} @ {fps} FPS)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = color_for_id(tr.track_id)

            if tr.time_since_update == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                # occluded: dashed predicted box
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tr.track_id} {tr.class_name} ({cx},{cy})"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if trail and tr.history:
                pts = tr.history[-trail:]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], color, 2)

        writer.write(frame)
        if show:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. {frame_count} frames -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-vehicle tracking via Kalman filter")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output_tracked.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--trail", type=int, default=0, help="Trajectory trail length (0=off)")
    parser.add_argument("--no-show", action="store_true", help="Disable live preview")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, conf=args.conf,
                 trail=args.trail, show=not args.no_show)


if __name__ == "__main__":
    main()