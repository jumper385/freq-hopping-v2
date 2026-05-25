import numpy as np


class ToneLayout:
    """
    Computes and exposes the boolean masks for an OFDM symbol's tone layout.

    All masks are over the *active* (non-guard) slice of the half-spectrum.

    Parameters
    ----------
    n_tones      : int   – FFT half-size (seq_len // 2)
    guard_pad    : int   – zero-pad tones at each edge
    pilot_spacing: int   – every N-th active tone is a pilot
    center_guard : int   – DC-guard half-width (zeroed either side of DC)
    """

    def __init__(self, n_tones: int, guard_pad: int, pilot_spacing: int, center_guard: int):
        self.n_tones = n_tones
        self.guard_pad = guard_pad
        self.pilot_spacing = pilot_spacing
        self.center_guard = center_guard

        active_len = n_tones - 2 * guard_pad
        center = active_len // 2

        center_mask = np.zeros(active_len, dtype=bool)
        center_mask[center - center_guard: center + center_guard] = True

        pilot_mask = np.zeros(active_len, dtype=bool)
        pilot_mask[::pilot_spacing] = True
        pilot_mask[center_mask] = False

        self.active_len = active_len
        self.center_mask: np.ndarray = center_mask
        self.pilot_mask: np.ndarray = pilot_mask
        self.data_mask: np.ndarray = ~pilot_mask & ~center_mask

        n_pilots = int(np.count_nonzero(pilot_mask))
        self.pilot_refs = np.array(
            [1 + 1j if i % 2 == 0 else -1 - 1j for i in range(n_pilots)],
            dtype=complex,
        )
        self.pilot_idx: np.ndarray = np.where(pilot_mask)[0]
        self.max_data_tones: int = int(np.count_nonzero(self.data_mask))


class OFDMModulator:
    """
    Converts data symbols to/from a time-domain OFDM burst (FFT half-spectrum).

    Responsibilities:
      - insert data symbols + pilots into the correct tones
      - IFFT → time domain
      - extract data tones from a received frequency-domain vector

    Does NOT handle preambles, CP, or power scaling – those live in the pipeline.

    Parameters
    ----------
    layout : ToneLayout
    """

    def __init__(self, layout: ToneLayout):
        self.layout = layout

    def modulate(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        data_symbols (complex, length ≤ max_data_tones) → full FFT vector
        (length = n_tones, zero-padded guard bins included).
        """
        lo = self.layout
        if len(data_symbols) > lo.max_data_tones:
            raise ValueError(
                f"data_symbols length {len(data_symbols)} exceeds "
                f"max_data_tones {lo.max_data_tones}"
            )
        tones = np.zeros(lo.active_len, dtype=complex)
        pad = np.zeros(lo.max_data_tones - len(data_symbols), dtype=complex)
        tones[lo.data_mask] = np.concatenate([data_symbols, pad])
        tones[lo.pilot_mask] = lo.pilot_refs

        full = np.concatenate([
            np.zeros(lo.guard_pad, dtype=complex),
            tones,
            np.zeros(lo.guard_pad, dtype=complex),
        ])
        return full  # length == n_tones

    def demodulate(self, freq_vector: np.ndarray) -> np.ndarray:
        """
        freq_vector (length == n_tones) → data tones only (complex array).
        No equalization – call PilotEqualizer afterwards.
        """
        lo = self.layout
        active = freq_vector[lo.guard_pad: lo.guard_pad + lo.active_len]
        return active  # caller picks data_mask / pilot_mask as needed
