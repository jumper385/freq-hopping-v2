"""
pipeline.py – wires the DSP building blocks into Tx and Rx paths.

Usage
-----
    from src.bb.pipeline import TxPipeline, RxPipeline, build_default_pipelines

    tx, rx = build_default_pipelines()
    frame   = tx.encode(b"hello world")
    msgs    = rx.decode(received_samples)
"""

import numpy as np

from src.utils.preambles import zadoff_chu
from src.bb.framing import Framer
from src.bb.mapping import QAMMapper
from src.bb.ofdm import ToneLayout, OFDMModulator
from src.bb.sync import PreambleSync
from src.bb.equalizer import PilotEqualizer


# ---------------------------------------------------------------------------
# Transmit pipeline
# ---------------------------------------------------------------------------

class TxPipeline:
    """
    bytes → complex baseband frame

    Stages:  Framer → QAMMapper → OFDMModulator → preamble + CP prepend
    """

    def __init__(
        self,
        framer: Framer,
        mapper: QAMMapper,
        modulator: OFDMModulator,
        preamble: np.ndarray,
        cp_len: int,
        tx_scale: float = 2**14,
        tile: int = 24,
    ):
        self.framer = framer
        self.mapper = mapper
        self.modulator = modulator
        self.preamble = preamble
        self.cp_len = cp_len
        self.tx_scale = tx_scale
        self.tile = tile

    @property
    def max_payload_bytes(self) -> int:
        lo = self.modulator.layout
        max_bits = lo.max_data_tones * self.mapper.N
        return (max_bits // 8) - self.framer.OVERHEAD

    def encode(self, payload: bytes) -> np.ndarray:
        """payload → single tiled Tx frame ready to feed to SDR.transmit()."""
        packet = self.framer.pack(payload)
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))

        lo = self.modulator.layout
        max_bits = lo.max_data_tones * self.mapper.N
        if len(bits) > max_bits:
            raise ValueError(
                f"Packet needs {len(bits)} bits but frame holds {max_bits}"
            )
        pad = np.zeros(max_bits - len(bits), dtype=np.uint8)
        bits = np.concatenate([bits, pad])

        data_symbols = self.mapper.map(bits)
        freq_vector = self.modulator.modulate(data_symbols)

        time_domain = np.fft.ifft(freq_vector)
        time_domain = _power_match(self.preamble, time_domain)
        cp = time_domain[-self.cp_len:]

        frame = np.concatenate([self.preamble, self.preamble, cp, time_domain])
        return np.tile(frame * self.tx_scale, self.tile)


# ---------------------------------------------------------------------------
# Receive pipeline
# ---------------------------------------------------------------------------

class RxPipeline:
    """
    complex baseband samples → list[bytes | None]

    Stages:  PreambleSync → FFT → PilotEqualizer → QAMMapper → Framer
    """

    def __init__(
        self,
        sync: PreambleSync,
        modulator: OFDMModulator,
        equalizer: PilotEqualizer,
        mapper: QAMMapper,
        framer: Framer,
    ):
        self.sync = sync
        self.modulator = modulator
        self.equalizer = equalizer
        self.mapper = mapper
        self.framer = framer

    def decode(self, rx: np.ndarray) -> list[bytes | None]:
        """
        Returns one entry per detected frame.  Entry is None when framing /
        CRC fails.
        """
        snippets = self.sync.find_frames(rx)
        if not snippets:
            return []

        freq = np.fft.fft(np.array(snippets))          # (n_frames, n_tones)
        active = np.array([
            self.modulator.demodulate(freq[i]) for i in range(len(snippets))
        ])                                               # (n_frames, active_len)
        data_symbols = self.equalizer.equalize(active)  # (n_frames, max_data_tones)

        results = []
        for sym in data_symbols:
            bits = self.mapper.demap(sym)
            raw = np.packbits(bits).tobytes()
            results.append(self.framer.unpack(raw))
        return results

    def decode_averaged(self, rx: np.ndarray) -> bytes | None:
        """
        Average all detected symbols before demapping (improves SNR when
        the same frame is repeated many times in the buffer).
        """
        snippets = self.sync.find_frames(rx)
        if not snippets:
            return None

        freq = np.fft.fft(np.array(snippets))
        active = np.array([self.modulator.demodulate(freq[i]) for i in range(len(snippets))])
        data_symbols = self.equalizer.equalize(active)

        avg = np.mean(data_symbols, axis=0)
        bits = self.mapper.demap(avg)
        raw = np.packbits(bits).tobytes()
        return self.framer.unpack(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pipelines(
    seq_len: int = 1024,
    pre_len: int = 32,
    cp_len: int = 32,
    guard_pad: int = 16,
    pilot_spacing: int = 8,
    center_guard: int = 32,
    qam_bits: int = 4,
    enable_cfo: bool = True,
    tx_scale: float = 2**14,
    tile: int = 24,
) -> tuple["TxPipeline", "RxPipeline"]:
    """
    Construct matched Tx/Rx pipelines from a shared parameter set.
    All structural objects (ToneLayout, preamble, …) are shared by reference
    so the two pipelines are guaranteed to be consistent.
    """
    preamble = zadoff_chu(pre_len, pre_len - 1)
    layout = ToneLayout(seq_len // 2, guard_pad, pilot_spacing, center_guard)
    modulator = OFDMModulator(layout)
    equalizer = PilotEqualizer(layout)
    framer = Framer()
    mapper = QAMMapper(qam_bits)
    sync = PreambleSync(preamble, cp_len, seq_len, enable_cfo=enable_cfo)

    tx = TxPipeline(framer, mapper, modulator, preamble, cp_len, tx_scale, tile)
    rx = RxPipeline(sync, modulator, equalizer, mapper, framer)
    return tx, rx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _power_match(ref_sig: np.ndarray, target_sig: np.ndarray) -> np.ndarray:
    ref_power = np.sqrt(np.mean(np.abs(ref_sig) ** 2))
    target_power = np.sqrt(np.mean(np.abs(target_sig) ** 2))
    scaled = target_sig * 3 * (ref_power / target_power)
    return scaled / np.max(np.abs(scaled))
