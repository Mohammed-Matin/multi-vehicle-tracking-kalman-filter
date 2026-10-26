import os

from ultralytics import YOLO

# COCO vehicle class ids
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


def _resolve_model(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cand = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    if os.path.exists(cand):
        return cand
    return path


class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt", conf_thresh=0.4):
        self.model = YOLO(_resolve_model(model_path))
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """Run detection; return list of dicts: {bbox:[x1,y1,x2,y2], class, conf}."""
        results = self.model(frame, verbose=False)[0]
        dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id in VEHICLE_CLASSES and conf >= self.conf_thresh:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class": VEHICLE_CLASSES[cls_id],
                    "conf": conf,
                })
        return dets