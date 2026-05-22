import numpy as np


class AdaptivePeakDetection:
    def __init__(self, std_thresh=4, style="std"):
        self.std_thresh = 4
        self.style = style

    def get_thresh(self, x):
        if self.style == "std":
            return self._get_std_thresh(x)

        if self.style == "peak":
            return self._get_peak_thresh(x)

    def _get_peak_thresh(self, x):
        x_min = x.min()
        x_max = x.max()
        return (x_max - x_min)/2

    def _get_std_thresh(self, x):
        u = np.mean(x)
        s = np.std(x)
        thresh = u + s * self.std_thresh
        return thresh
