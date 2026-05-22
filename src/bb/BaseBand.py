import numpy as np
from scipy.interpolate import interp1d

from src.utils.preambles import zadoff_chu
from src.utils.AdaptivePeakDetection import AdaptivePeakDetection

import logging
log = logging.getLogger(__name__)


class BaseBand:

    def __init__(self, seq_len=1024, pre_len=32, _enable_cfo=True):
        self.seq_len = seq_len
        self.pre_len = pre_len
        self._rx_pre_margin = int(self.pre_len * 0.1)
        self.total_len = self.seq_len + 2 * self.pre_len

        self.preamble = zadoff_chu(self.pre_len, self.pre_len - 1)

        self._enable_cfo = _enable_cfo

    def _encode_symbol(self, symbol, dtype=complex, pilot_spacing=8, guard_pad=16):
        n_tones = int(self.seq_len / 2)

        tones = np.zeros(n_tones - 2 * guard_pad, dtype=complex)
        pilot_mask = np.zeros(n_tones - 2 * guard_pad, dtype=bool)
        pilot_mask[::pilot_spacing] = True

        max_data_len = np.count_nonzero(~pilot_mask)
        data_len = len(symbol)

        zero_pad = np.zeros(max_data_len - data_len)
        tones[~pilot_mask] = np.concat([symbol, zero_pad])
        tones[pilot_mask] = 1 + 1j

        return np.concat([np.zeros(guard_pad, dtype=complex),
                          tones,
                          np.zeros(guard_pad, dtype=complex)])

    def gen_tx(self):
        data_fft = self._encode_symbol([0, 0.5+0.5j, 1+1j, 0.5+0.5j, 0, -0.5-0.5j, -1-1j, -0.5-0.5j, 0])
        data_seq = np.fft.ifft(data_fft)

        out = np.concatenate([self.preamble, self.preamble, data_seq])
        return out

    def equalize_symbols(self, symbols, guard_pad=16, pilot_spacing=8):
        n_tones = int(self.seq_len / 2)
        ref_pilot = np.zeros(n_tones - 2 * guard_pad, dtype=complex)
        pilot_mask = np.zeros(n_tones - 2 * guard_pad, dtype=bool)
        pilot_mask[::pilot_spacing] = True

        ref_pilot[pilot_mask] = 1 + 1j

        active = symbols[:, guard_pad:-guard_pad]
        sym_pilots = active[:, pilot_mask]

        pilot_idx = np.where(pilot_mask)[0]
        all_idx = np.arange(active.shape[1])

        H_pilots = sym_pilots / (1 + 1j)

        equalized = np.zeros_like(active)
        for i in range(active.shape[0]):
            f_real = interp1d(pilot_idx, H_pilots[i].real, kind='linear', fill_value='extrapolate')
            f_imag = interp1d(pilot_idx, H_pilots[i].imag, kind='linear', fill_value='extrapolate')
            H = f_real(all_idx) + 1j * f_imag(all_idx)
            equalized[i] = active[i] / H

        return equalized

    def det_rx(self, rx):
        """
        finds ALL possible frames in the rx array
        """
        symbols = self._find_snippets(rx)
        if len(symbols) < 1:
            return []

        out = np.fft.fft(symbols)
        out = self.equalize_symbols(out)
        print(out.shape)

        return out

    def _get_threshold(self, corr_mag):
        corr_apd = AdaptivePeakDetection(style="peak")
        corr_threshold = corr_apd.get_thresh(corr_mag)
        return corr_threshold

    def _get_preambles(self, corr_mag, corr_threshold):
        mask = corr_mag > corr_threshold
        rising_edges = np.where(np.diff(mask.astype(int)) > 0)[0]
        return rising_edges

    def _get_freq_offset(self, rx, pre1, pre2):
        out = np.angle(np.dot(pre1, np.conj(pre2)))
        out = out / (2 * np.pi * self.pre_len/(self.seq_len))
        return out

    def _find_snippets(self, rx):
        """
        finds snippets in the received signal
        assumes preamble exists
        gives you all snippets except the last one int he frame
        cos last one could be prematurely snipped off
        """
        corr = np.correlate(rx, self.preamble)
        corr_mag = np.abs(corr)
        corr_threshold = self._get_threshold(corr_mag)

        rising_edges = self._get_preambles(corr_mag, corr_threshold)

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

                # phase angle of preamble corr
                pre1 = rx[rising_edges[idx]:rising_edges[idx]+self.pre_len]
                pre2 = rx[rising_edges[idx+1]:rising_edges[idx+1]+self.pre_len]
                cfo = self._get_freq_offset(rx, pre1, pre2)

                snippet = rx[start_idx:end_idx]
                if self._enable_cfo:
                    t = np.arange(0, len(snippet))
                    snippet *= np.exp(-2j * np.pi * t * -1 * cfo / self.seq_len)
                symbols.append(snippet)

        return symbols

    def det_dbg(self, rx):
        corr_mag = np.abs(np.correlate(rx, self.preamble))

        return corr_mag
