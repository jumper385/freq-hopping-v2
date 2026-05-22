import numpy as np

from src.utils.preambles import zadoff_chu
from src.utils.AdaptivePeakDetection import AdaptivePeakDetection

import logging
log = logging.getLogger(__name__)


class BaseBand:

    def __init__(self, seq_len=1024, pre_len=32):
        self.seq_len = seq_len
        self.pre_len = pre_len
        self._rx_pre_margin = int(self.pre_len * 0.1)
        self.total_len = self.seq_len + 2 * self.pre_len

        self.preamble = zadoff_chu(self.pre_len, self.pre_len - 1)
        pass

    def gen_tx(self):
        t = np.linspace(0, 1, self.seq_len)
        x = np.sin(2 * np.pi * 10 * t) / 100
        y = np.sin(2 * np.pi * 10 * t) / 100
        for i in range(10, 100):
            x += np.sin(2 * np.pi * i*2 * t) / 100
            y += np.sin(2 * np.pi * i*2 * t) / 100

        out = np.concatenate([self.preamble, self.preamble, x + 1j * y])
        return out

    def det_rx(self, rx):
        """
        finds ALL possible frames in the rx array
        """
        symbols = self._find_snippets(rx)
        if len(symbols) < 1:
            print("No symbols found...")
            return []

        return symbols

    def _get_threshold(self, corr_mag):
        corr_apd = AdaptivePeakDetection(style="peak")
        corr_threshold = corr_apd.get_thresh(corr_mag)
        return corr_threshold

    def _get_preambles(self, corr_mag, corr_threshold):
        mask = corr_mag > corr_threshold
        rising_edges = np.where(np.diff(mask.astype(int)) > 0)[0]
        return rising_edges

    def _find_snippets(self, rx):
        """
        finds snippets in the received signal
        assumes preamble exists
        gives you all snippets except the last one int he frame
        cos last one could be prematurely snipped off
        """
        corr_mag = np.abs(np.correlate(rx, self.preamble))
        corr_threshold = self._get_threshold(corr_mag)
        print(corr_threshold)

        rising_edges = self._get_preambles(corr_mag, corr_threshold)
        print(rising_edges)

        if len(rising_edges) < 2:
            print("Couldnt find sufficient number of preambles")
            return np.array([])

        symbols = []
        for idx, _ in enumerate(rising_edges[:-2]):
            spacing = rising_edges[idx+1] - rising_edges[idx]
            min_margin = self.pre_len - self._rx_pre_margin
            max_margin = self.pre_len + self._rx_pre_margin

            if min_margin < spacing < max_margin:
                # +1 to snip off the final bit of preamble
                # NOTE: likely pre-emptive threshold detection
                start_idx = rising_edges[idx+1] + self.pre_len + 1
                end_idx = rising_edges[idx+2]  # get all up to next pre
                snippet = rx[start_idx:end_idx]
                symbols.append(snippet)

        return symbols

    def det_dbg(self, rx):
        corr_mag = np.abs(np.correlate(rx, self.preamble))

        return corr_mag
