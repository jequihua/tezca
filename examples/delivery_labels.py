"""Tiny synthetic custom model for delivery.py; stable backend name across entry points."""

import numpy as np


class SignLabels:
    def fit_predict(self, frame):
        self.threshold = float(np.mean(frame[:, 0]))
        return self.predict(frame)

    def predict(self, frame):
        return np.array([-1 if value == self.threshold else "low" if value < self.threshold else "high"
                         for value in frame[:, 0]], dtype=object)
