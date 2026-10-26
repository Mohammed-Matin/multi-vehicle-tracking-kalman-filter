import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import KalmanFilterTracker


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / float(union + 1e-6)


class Track:
    def __init__(self, track_id, bbox, class_name, confirm_frames=3):
        self.track_id = track_id
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.kf = KalmanFilterTracker(cx, cy)
        self.bbox = [float(x1), float(y1), float(x2), float(y2)]
        self.w = x2 - x1
        self.h = y2 - y1
        self.class_name = class_name
        self.hits = 1
        self.time_since_update = 0
        self.history = [(int(cx), int(cy))]
        self.confirm_frames = confirm_frames
        self.confirmed = self.hits >= confirm_frames

    @property
    def is_confirmed(self):
        return self.hits >= self.confirm_frames

    def predict(self):
        px, py = self.kf.predict()
        self.bbox = [px - self.w / 2.0, py - self.h / 2.0,
                     px + self.w / 2.0, py + self.h / 2.0]
        self.time_since_update += 1
        self.history.append((int(px), int(py)))
        if len(self.history) > 30:
            self.history.pop(0)
        return self.bbox

    def update(self, bbox, class_name):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.kf.update(cx, cy)
        self.w = x2 - x1
        self.h = y2 - y1
        self.bbox = [float(x1), float(y1), float(x2), float(y2)]
        self.class_name = class_name
        self.hits += 1
        self.time_since_update = 0
        self.confirmed = self.hits >= self.confirm_frames


class MultiVehicleTracker:
    def __init__(self, max_age=20, mahal_thresh=9.49,
                 confirm_frames=3, tentative_age=5):
        self.max_age = max_age
        self.mahal_thresh = mahal_thresh
        self.confirm_frames = confirm_frames
        self.tentative_age = tentative_age
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        # 1. Predict all tracks
        for t in self.tracks:
            t.predict()

        num_t = len(self.tracks)
        num_d = len(detections)

        matched_t, matched_d = [], []
        if num_t > 0 and num_d > 0:
            cost = np.zeros((num_t, num_d), dtype=np.float32)
            for ti, tr in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    cost[ti, di] = tr.kf.mahalanobis(cx, cy)

            # Hungarian on cost
            ti_idx, di_idx = linear_sum_assignment(cost)
            for ti, di in zip(ti_idx, di_idx):
                if cost[ti, di] <= self.mahal_thresh:
                    matched_t.append(ti)
                    matched_d.append(di)
                    self.tracks[ti].update(detections[di]["bbox"],
                                          detections[di]["class"])

        unmatched_t = [i for i in range(num_t) if i not in matched_t]
        unmatched_d = [i for i in range(num_d) if i not in matched_d]

        # 2. New tracks from unmatched detections
        for di in unmatched_d:
            new_t = Track(self.next_id, detections[di]["bbox"],
                          detections[di]["class"], self.confirm_frames)
            self.tracks.append(new_t)
            self.next_id += 1

        # 3. Age / delete
        alive = []
        for ti, tr in enumerate(self.tracks):
            if tr.time_since_update == 0:
                alive.append(tr)
                continue
            # tentative tracks die faster
            age_limit = self.tentative_age if not tr.is_confirmed else self.max_age
            if tr.time_since_update <= age_limit:
                alive.append(tr)
        self.tracks = alive

        return self.tracks