"""Generate PDF report for Multi-Vehicle Kalman Tracking project."""
import os
import cv2
from fpdf import FPDF

# Paths
REPO_DIR = "/home/matin/laboratery/AIR/kalmanFilter/multi-vehicle-tracking-kalman-filter"
TRACKED_DIR = "/mnt/d/WSL/Videos/AIR/tracked"
OUT_PDF = "/mnt/d/WSL/Videos/AIR/MultiVehicle_Tracking_Report.pdf"

tracked_files = sorted([f for f in os.listdir(TRACKED_DIR) if f.endswith("_tracked.mp4")])
print("Tracked files:", tracked_files)

# Extract frames from first two videos
frame_paths = []
for vf in tracked_files[:2]:
    vp = os.path.join(TRACKED_DIR, vf)
    cap = cv2.VideoCapture(vp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.3))
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        scale = 700 / w
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        safe = "".join(c if c.isalnum() else "_" for c in os.path.splitext(vf)[0])[:30]
        fp = os.path.join(REPO_DIR, f"frame_{safe}.jpg")
        cv2.imwrite(fp, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_paths.append((vf, fp, new_w, new_h))
        print(f"Extracted frame: {fp} ({new_w}x{new_h})")
    cap.release()


def sanitize(text):
    """Replace characters outside latin-1 with '?'."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def read_file(rel):
    p = os.path.join(REPO_DIR, rel)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            return sanitize(fh.read())
    return "[File not found: " + rel + "]"


code_files = [
    "vehicle_tracker.py",
    "tracker/kalman.py",
    "tracker/detector.py",
    "tracker/association.py",
    "utils/video_io.py",
    "requirements.txt",
    "README.md",
]
file_contents = {f: read_file(f) for f in code_files}

folder_tree = """multi-vehicle-tracking-kalman-filter/
|-- vehicle_tracker.py          # CLI entry point
|-- requirements.txt            # Python dependencies
|-- README.md                   # Project overview
|-- download_videos.sh          # Script to fetch sample videos
|-- tracker/
|   |-- __init__.py
|   |-- kalman.py               # Constant-velocity Kalman filter + Mahalanobis distance
|   |-- detector.py             # YOLOv8 vehicle detector wrapper
|   '-- association.py          # Track manager: Hungarian matching, IoU backstop, merging
|-- utils/
|   |-- __init__.py
|   '-- video_io.py             # Video open/validate/write helpers
|-- samples/                    # Small test clips (gitignored)
|   |-- car-detection.mp4
|   '-- person-bicycle-car-detection.mp4
|-- models/                     # YOLO weights (yolov8n.pt, downloaded at runtime)
`-- output/                     # Generated annotated videos (local only)
"""


class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, "Multi-Vehicle Tracking Using Kalman Filter", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("helvetica", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "Python CLI Utility  |  OpenCV + YOLOv8 + SciPy  |  Lab Assignment Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}  |  Mohammed-Matin  |  GitHub: github.com/Mohammed-Matin/multi-vehicle-tracking-kalman-filter", align="C")


def section_title(pdf, text):
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)


def body_text(pdf, text, size=10):
    pdf.set_font("helvetica", "", size)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, sanitize(text))
    pdf.ln(2)


def code_block(pdf, title, code):
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("courier", "", 7)
    pdf.set_text_color(20, 20, 20)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(200, 200, 200)
    code_w = pdf.w - pdf.l_margin - pdf.r_margin
    lines = code.split("\n")
    for ln in lines:
        while len(ln) > 110:
            pdf.multi_cell(code_w, 3, sanitize(ln[:110]), fill=True)
            ln = ln[110:]
        pdf.multi_cell(code_w, 3, sanitize(ln), fill=True)
    pdf.ln(2)


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# 1. Title
pdf.set_font("helvetica", "B", 20)
pdf.set_text_color(30, 60, 120)
pdf.cell(0, 15, "Multi-Vehicle Tracking Using Kalman Filter", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(2)
pdf.set_font("helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Python Command-Line Utility for Vehicle Detection & Tracking in Video", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(2)
pdf.set_font("helvetica", "", 10)
pdf.cell(0, 6, "Author: Mohammed-Matin", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.cell(0, 6, "GitHub: https://github.com/Mohammed-Matin/multi-vehicle-tracking-kalman-filter", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(8)

# 2. Problem Description
section_title(pdf, "1. Problem Description")
body_text(pdf,
    "Develop a general-purpose, multi-vehicle tracking system that reads any custom input video "
    "(specified via terminal), detects vehicles in each frame, assigns a unique ID to every vehicle, "
    "and tracks their movement across consecutive frames using a Kalman Filter. The system must handle "
    "occlusions (temporary missing detections), create new tracks for entering vehicles, remove lost "
    "tracks, and output an annotated video with bounding boxes, track IDs, vehicle classes, and positions.")

# 3. Folder Structure
section_title(pdf, "2. Folder Structure")
body_text(pdf, "The project is organized as a small CLI utility (not a backend/frontend web app):")
pdf.set_font("courier", "", 7)
pdf.set_fill_color(245, 245, 245)
tree_w = pdf.w - pdf.l_margin - pdf.r_margin
pdf.set_x(pdf.l_margin)
for ln in folder_tree.split("\n"):
    pdf.multi_cell(tree_w, 3, sanitize(ln), fill=True)
pdf.ln(3)

# 4. GitHub Repo Link
section_title(pdf, "3. GitHub Repository")
pdf.set_font("helvetica", "", 10)
pdf.set_text_color(0, 80, 160)
pdf.cell(0, 6, "https://github.com/Mohammed-Matin/multi-vehicle-tracking-kalman-filter", new_x="LMARGIN", new_y="NEXT", link="https://github.com/Mohammed-Matin/multi-vehicle-tracking-kalman-filter")
pdf.ln(2)
body_text(pdf, "Public repository containing all source code, configuration, and documentation. Sample videos and model weights are gitignored and downloaded/located at runtime.")

# 5. Approach Explanation
section_title(pdf, "4. Approach Explanation")
body_text(pdf,
    "The pipeline follows the tracking-by-detection paradigm used in MATLAB's Motion-Based Multiple "
    "Object Tracking tutorial and modern MOT methods (SORT/DeepSORT):")
pdf.set_font("helvetica", "B", 10)
pdf.cell(0, 6, "A. Video I/O (utils/video_io.py):", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", "", 10)
body_text(pdf, "Accepts input path via CLI (argparse). Validates file existence with os.path.exists(). "
               "Reads FPS, width, height from the video container via cv2.VideoCapture properties. "
               "Writes annotated frames to output via cv2.VideoWriter (mp4v codec).")
pdf.ln(1)

pdf.set_font("helvetica", "B", 10)
pdf.cell(0, 6, "B. Vehicle Detection (tracker/detector.py):", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", "", 10)
body_text(pdf, "YOLOv8n (nano) runs on every frame. Detections are filtered to COCO vehicle classes: "
               "Car (2), Motorcycle (3), Bus (5), Truck (7). Output: bounding boxes [x1,y1,x2,y2], "
               "confidence, class label.")
pdf.ln(1)

pdf.set_font("helvetica", "B", 10)
pdf.cell(0, 6, "C. Kalman Filter (tracker/kalman.py):", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", "", 10)
body_text(pdf, "Constant-Velocity (CV) model per tracked vehicle. State vector X = [x, y, vx, vy]^T "
               "(center position + pixel velocity). Prediction uses x_k = x_{k-1} + vx·dt. Process noise Q "
               "absorbs small accelerations; measurement noise R handles detector jitter. During occlusion, "
               "the filter predicts position without a measurement update.")
pdf.ln(1)

pdf.set_font("helvetica", "B", 10)
pdf.cell(0, 6, "D. Data Association & Track Management (tracker/association.py):", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", "", 10)
body_text(pdf, "Mahalanobis distance between each track's predicted state and new detections (covariance-aware). "
               "Hungarian algorithm (scipy.optimize.linear_sum_assignment) solves the assignment. Matches below "
               "threshold are paired; unmatched detections spawn new tracks. Two safeguards prevent track "
               "fragmentation: (1) IoU backstop — an unmatched detection overlapping an existing track is "
               "assigned to it instead of creating a duplicate ID; (2) track merging — overlapping tracks are "
               "collapsed, keeping the oldest/most-confirmed. Tracks unconfirmed for N frames are dropped "
               "(confirmation gate), and tracks missing for max_age frames are removed.")
pdf.ln(1)

pdf.set_font("helvetica", "B", 10)
pdf.cell(0, 6, "E. Visualization & Output (vehicle_tracker.py):", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", "", 10)
body_text(pdf, "Each tracked vehicle gets a unique color (hashed from track ID). Bounding box is drawn "
               "(solid when detected, dashed when occluded/predicted). Label shows ID: Class | (cx, cy). "
               "Optional trajectory trail connects recent center points. Output video preserves original "
               "FPS and resolution.")

# 6. File Code
section_title(pdf, "5. Source Code")
for f in code_files:
    pdf.add_page()
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 7, f"File: {f}", new_x="LMARGIN", new_y="NEXT")
    code_block(pdf, f"Contents of {f}", file_contents[f])

# 7. Video Outputs (first two videos with frames)
pdf.add_page()
section_title(pdf, "6. Video Outputs (Sample Frames)")
body_text(pdf, "The following frames are extracted from the first two processed output videos "
               "(stored locally in D:\\WSL\\Videos\\AIR\\tracked\\). Full annotated videos are MP4 files "
               "and are not embedded in this PDF.")
pdf.ln(3)

for vf, fp, nw, nh in frame_paths:
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, f"Video: {vf}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    pdf.cell(0, 5, f"Resolution at extraction: {nw}x{nh}  |  Source: {os.path.join(TRACKED_DIR, vf)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    # Aspect-fit image to ~180mm width
    max_w = 180
    scale = max_w / nw
    disp_w = nw * scale
    disp_h = nh * scale
    if disp_h > 140:
        scale = 140 / nh
        disp_w = nw * scale
        disp_h = nh * scale
    x = (pdf.w - disp_w) / 2
    pdf.image(fp, x=x, w=disp_w, h=disp_h)
    pdf.ln(4)

pdf.output(OUT_PDF)
print("PDF generated:", OUT_PDF)
print("Size:", os.path.getsize(OUT_PDF), "bytes")