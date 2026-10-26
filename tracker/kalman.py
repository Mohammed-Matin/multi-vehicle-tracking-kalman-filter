import numpy as np
import cv2


class KalmanFilterTracker:
    """Constant-velocity Kalman filter for a single vehicle (center x, y).

    State vector: X = [x, y, vx, vy]^T
    Measurement:  Z = [x, y]^T
    """

    def __init__(self, init_x: float, init_y: float, dt: float = 1.0):
        self.dt = dt
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition: constant velocity
        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,       0],
            [0, 0, 0,       1],
        ], dtype=np.float32)

        # We only measure x, y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise (accounts for small accel changes)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        # Measurement noise (detector jitter)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.array(
            [[init_x], [init_y], [0.0], [0.0]], dtype=np.float32
        )

    def predict(self):
        """Predict next position. Returns (x, y)."""
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def update(self, meas_x: float, meas_y: float):
        """Correct state with a new detection."""
        meas = np.array([[np.float32(meas_x)], [np.float32(meas_y)]])
        self.kf.correct(meas)

    @property
    def position(self):
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    @property
    def velocity(self):
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])