import torch
import numpy as np
from sionna.phy.mapping import Mapper, Demapper


class QAMMapper:
    """
    Wraps Sionna QAM map/demap.  Stateless with respect to payload content.

    Parameters
    ----------
    N : int
        Bits per symbol (e.g. 4 → 16-QAM, 6 → 64-QAM).
    """

    def __init__(self, N: int):
        self.N = N
        self._mapper = Mapper("qam", N)
        self._demapper = Demapper("app", "qam", N, hard_out=True)

    def map(self, bits: np.ndarray) -> np.ndarray:
        """bits (uint8 flat array) → complex symbols (numpy)."""
        t = torch.tensor(bits, dtype=torch.float32)
        return self._mapper(t).cpu().numpy()

    def demap(self, symbols: np.ndarray) -> np.ndarray:
        """complex symbols → uint8 bit array (hard decision)."""
        t = torch.tensor(symbols)
        bits = self._demapper(t, no=0.01)
        return bits.int().cpu().numpy().astype(np.uint8)
