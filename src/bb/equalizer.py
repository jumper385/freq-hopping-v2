import numpy as np
from scipy.interpolate import interp1d
from src.bb.ofdm import ToneLayout


class PilotEqualizer:
    """
    Single-tap frequency-domain equalizer driven by known pilot tones.

    For each received OFDM symbol the equalizer:
    1. Extracts pilots and divides by the reference values to estimate H.
    2. Linearly interpolates H across all active tones.
    3. Divides each tone by the interpolated channel estimate.
    4. Returns only the data tones (pilots and DC-guard are discarded).

    Parameters
    ----------
    layout : ToneLayout
    """

    def __init__(self, layout: ToneLayout):
        self.layout = layout

    def equalize(self, active_symbols: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        active_symbols : np.ndarray, shape (n_symbols, active_len)
            Frequency-domain active slice (guard bins already removed).

        Returns
        -------
        np.ndarray, shape (n_symbols, max_data_tones)
        """
        lo = self.layout
        sym_pilots = active_symbols[:, lo.pilot_mask]
        H_pilots = sym_pilots / lo.pilot_refs[np.newaxis, :]

        all_idx = np.arange(lo.active_len)
        equalized = np.zeros_like(active_symbols)

        for i in range(active_symbols.shape[0]):
            f_real = interp1d(
                lo.pilot_idx, H_pilots[i].real, kind="linear", fill_value="extrapolate"
            )
            f_imag = interp1d(
                lo.pilot_idx, H_pilots[i].imag, kind="linear", fill_value="extrapolate"
            )
            H = f_real(all_idx) + 1j * f_imag(all_idx)
            equalized[i] = active_symbols[i] / H

        return equalized[:, lo.data_mask]
