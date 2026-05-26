"""
Tests for src/bb/equalizer.py  (PilotEqualizer)
"""
import pytest
import numpy as np
from src.bb.ofdm import ToneLayout, OFDMModulator
from src.bb.equalizer import PilotEqualizer


@pytest.fixture
def layout():
    return ToneLayout(n_tones=512, guard_pad=16, pilot_spacing=8, center_guard=32)


@pytest.fixture
def equalizer(layout):
    return PilotEqualizer(layout)


@pytest.fixture
def modulator(layout):
    return OFDMModulator(layout)


class TestPilotEqualizer:
    def test_output_shape(self, equalizer, modulator, layout):
        n_sym = 3
        data = np.random.randn(layout.max_data_tones) + 1j * np.random.randn(layout.max_data_tones)
        freq_vecs = np.array([modulator.modulate(data) for _ in range(n_sym)])
        active = freq_vecs[:, layout.guard_pad: layout.guard_pad + layout.active_len]
        out = equalizer.equalize(active)
        assert out.shape == (n_sym, layout.max_data_tones)

    def test_flat_channel_passthrough(self, equalizer, modulator, layout):
        """H=1+1j everywhere; pilots are already 1+1j references so H estimate is exact."""
        data = np.random.randn(layout.max_data_tones) + 1j * np.random.randn(layout.max_data_tones)
        freq = modulator.modulate(data)
        # apply flat channel
        channel = 1 + 1j
        freq_rx = freq * channel

        active = freq_rx[layout.guard_pad: layout.guard_pad + layout.active_len][np.newaxis, :]
        out = equalizer.equalize(active)[0]
        assert np.allclose(out, data, atol=1e-6)

    def test_returns_data_tones_only(self, equalizer, modulator, layout):
        data = np.ones(layout.max_data_tones, dtype=complex)
        freq = modulator.modulate(data)
        active = freq[layout.guard_pad: layout.guard_pad + layout.active_len][np.newaxis, :]
        out = equalizer.equalize(active)
        # pilots and center guard bins should not appear
        assert out.shape[1] == layout.max_data_tones
