import numpy as np
from src.utils.AdaptivePeakDetection import AdaptivePeakDetection


class PreambleSync:
    """
    Detects OFDM frames in a received sample stream using a known preamble.

    The transmitter sends two consecutive copies of the preamble followed by
    a cyclic prefix then the OFDM symbol.  This block:

    1. Cross-correlates the RX stream with the preamble.
    2. Finds rising-edge threshold crossings (preamble candidates).
    3. Validates pairs whose spacing matches the preamble length (± margin).
    4. Optionally estimates and corrects CFO using the phase rotation between
       the two preamble copies.

    Parameters
    ----------
    preamble  : np.ndarray  – known preamble sequence (complex)
    cp_len    : int         – cyclic prefix length in samples
    seq_len   : int         – OFDM symbol length (FFT size)
    enable_cfo: bool        – apply CFO correction to extracted snippets
    margin    : int | None  – preamble-spacing tolerance; defaults to
                              max(1, int(len(preamble) * 0.1))
    """

    def __init__(
        self,
        preamble: np.ndarray,
        cp_len: int,
        seq_len: int,
        enable_cfo: bool = True,
        margin: int | None = None,
    ):
        self.preamble = preamble
        self.cp_len = cp_len
        self.seq_len = seq_len
        self.n_tones = seq_len // 2
        self.enable_cfo = enable_cfo
        pre_len = len(preamble)
        self.pre_len = pre_len
        self.margin = margin if margin is not None else max(1, int(pre_len * 0.1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_frames(self, rx: np.ndarray) -> list[np.ndarray]:
        """
        Returns a list of time-domain snippets (one per detected frame),
        each of length seq_len // 2 (n_tones), with CFO correction applied
        if enabled.
        """
        corr_mag = np.abs(np.correlate(rx, self.preamble))
        threshold = self._threshold(corr_mag)
        edges = self._rising_edges(corr_mag, threshold)

        if len(edges) < 2:
            return []

        snippets = []
        for i in range(len(edges) - 2):
            spacing = edges[i + 1] - edges[i]
            if not (self.pre_len - self.margin < spacing < self.pre_len + self.margin):
                continue

            # edges[i+1] is the last index where corr_mag <= threshold,
            # so the preamble actually starts at edges[i+1] + 1.
            start = edges[i + 1] + 1 + self.pre_len + self.cp_len
            end = start + self.n_tones
            if end > len(rx):
                continue

            snippet = rx[start:end].copy()
            if self.enable_cfo:
                cfo = self._cfo(rx, edges[i], edges[i + 1])
                t = np.arange(start, start + len(snippet))
                snippet *= np.exp(-2j * np.pi * t * (-cfo) / self.seq_len)

            snippets.append(snippet)

        return snippets

    def correlation_magnitude(self, rx: np.ndarray) -> np.ndarray:
        """Expose raw correlation for debugging / visualisation."""
        return np.abs(np.correlate(rx, self.preamble))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _threshold(self, corr_mag: np.ndarray) -> float:
        return AdaptivePeakDetection(style="peak").get_thresh(corr_mag)

    def _rising_edges(self, corr_mag: np.ndarray, threshold: float) -> np.ndarray:
        mask = corr_mag > threshold
        return np.where(np.diff(mask.astype(int)) > 0)[0]

    def _cfo(self, rx: np.ndarray, edge1: int, edge2: int) -> float:
        pre1 = rx[edge1: edge1 + self.pre_len]
        pre2 = rx[edge2: edge2 + self.pre_len]
        angle = np.angle(np.dot(pre1, np.conj(pre2)))
        return angle / (2 * np.pi * self.pre_len / self.seq_len)
