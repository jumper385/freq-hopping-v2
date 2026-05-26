"""
Tests for src/bb/ofdm.py  (ToneLayout + OFDMModulator)
"""
import pytest
import numpy as np
from src.bb.ofdm import ToneLayout, OFDMModulator


@pytest.fixture
def layout():
    return ToneLayout(n_tones=512, guard_pad=16, pilot_spacing=8, center_guard=32)


@pytest.fixture
def modulator(layout):
    return OFDMModulator(layout)


class TestToneLayout:
    def test_masks_cover_active_len(self, layout):
        total = layout.active_len
        covered = (layout.pilot_mask | layout.center_mask | layout.data_mask)
        assert covered.all()
        assert covered.sum() == total

    def test_masks_are_disjoint(self, layout):
        assert not (layout.pilot_mask & layout.center_mask).any()
        assert not (layout.pilot_mask & layout.data_mask).any()
        assert not (layout.center_mask & layout.data_mask).any()

    def test_max_data_tones_positive(self, layout):
        assert layout.max_data_tones > 0

    def test_pilot_refs_length(self, layout):
        assert len(layout.pilot_refs) == layout.pilot_mask.sum()

    def test_pilot_idx_matches_mask(self, layout):
        assert np.array_equal(layout.pilot_idx, np.where(layout.pilot_mask)[0])


class TestOFDMModulator:
    def test_modulate_output_length(self, modulator, layout):
        data = np.ones(layout.max_data_tones, dtype=complex)
        freq = modulator.modulate(data)
        assert len(freq) == layout.n_tones

    def test_modulate_guard_bins_zero(self, modulator, layout):
        data = np.ones(layout.max_data_tones, dtype=complex)
        freq = modulator.modulate(data)
        assert np.all(freq[:layout.guard_pad] == 0)
        assert np.all(freq[-layout.guard_pad:] == 0)

    def test_modulate_pilots_present(self, modulator, layout):
        data = np.zeros(layout.max_data_tones, dtype=complex)
        freq = modulator.modulate(data)
        active = freq[layout.guard_pad: layout.guard_pad + layout.active_len]
        pilots_in_freq = active[layout.pilot_mask]
        assert np.allclose(pilots_in_freq, layout.pilot_refs)

    def test_modulate_too_many_symbols_raises(self, modulator, layout):
        with pytest.raises(ValueError):
            modulator.modulate(np.ones(layout.max_data_tones + 1, dtype=complex))

    def test_modulate_partial_fill(self, modulator, layout):
        data = np.ones(10, dtype=complex)
        freq = modulator.modulate(data)
        assert len(freq) == layout.n_tones

    def test_demodulate_returns_active_slice(self, modulator, layout):
        freq = np.random.randn(layout.n_tones) + 1j * np.random.randn(layout.n_tones)
        active = modulator.demodulate(freq)
        assert len(active) == layout.active_len
