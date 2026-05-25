"""
Tests for src/bb/sync.py  (PreambleSync)
"""
import pytest
import numpy as np
from src.utils.preambles import zadoff_chu
from src.bb.sync import PreambleSync


def make_sync(pre_len=32, cp_len=16, seq_len=512, enable_cfo=False):
    preamble = zadoff_chu(pre_len, pre_len - 1)
    return PreambleSync(preamble, cp_len, seq_len, enable_cfo=enable_cfo)


def make_frame(sync: PreambleSync, payload_time: np.ndarray) -> np.ndarray:
    """Build a well-formed frame: pre + pre + CP + payload."""
    cp = payload_time[-sync.cp_len:]
    return np.concatenate([sync.preamble, sync.preamble, cp, payload_time])


class TestPreambleSync:
    def test_finds_single_frame(self):
        rng = np.random.default_rng(0)
        sync = make_sync()
        payload = rng.standard_normal(sync.n_tones) + 1j * rng.standard_normal(sync.n_tones)
        frame = make_frame(sync, payload)
        # embed in a larger buffer with leading/trailing silence
        rx = np.concatenate([np.zeros(50, dtype=complex), frame, np.zeros(50, dtype=complex)])
        snippets = sync.find_frames(rx)
        assert len(snippets) == 1

    def test_snippet_length(self):
        rng = np.random.default_rng(1)
        sync = make_sync()
        payload = rng.standard_normal(sync.n_tones) + 1j * rng.standard_normal(sync.n_tones)
        frame = make_frame(sync, payload)
        rx = np.concatenate([np.zeros(50, dtype=complex), frame, np.zeros(50, dtype=complex)])
        snippets = sync.find_frames(rx)
        assert snippets[0].shape == (sync.n_tones,)

    def test_snippet_content_matches_payload(self):
        rng = np.random.default_rng(2)
        sync = make_sync(enable_cfo=False)
        payload = rng.standard_normal(sync.n_tones) + 1j * rng.standard_normal(sync.n_tones)
        frame = make_frame(sync, payload)
        rx = np.concatenate([np.zeros(50, dtype=complex), frame, np.zeros(50, dtype=complex)])
        snippets = sync.find_frames(rx)
        assert np.allclose(snippets[0], payload)

    def test_no_preamble_returns_empty(self):
        sync = make_sync()
        # All-zero buffer: correlation is zero everywhere, adaptive threshold
        # never crossed, so no frames detected.
        rx = np.zeros(1000, dtype=complex)
        snippets = sync.find_frames(rx)
        assert snippets == []

    def test_correlation_magnitude_shape(self):
        sync = make_sync()
        rx = np.zeros(200, dtype=complex)
        mag = sync.correlation_magnitude(rx)
        expected_len = len(rx) - len(sync.preamble) + 1
        assert len(mag) == expected_len

    def test_multiple_frames(self):
        rng = np.random.default_rng(3)
        sync = make_sync()
        payload = rng.standard_normal(sync.n_tones) + 1j * rng.standard_normal(sync.n_tones)
        frame = make_frame(sync, payload)
        # two frames back-to-back with silence either side
        rx = np.concatenate([
            np.zeros(30, dtype=complex),
            frame,
            np.zeros(20, dtype=complex),
            frame,
            np.zeros(30, dtype=complex),
        ])
        snippets = sync.find_frames(rx)
        # At least one snippet per frame; adaptive threshold may detect
        # additional valid pairs near frame boundaries.
        assert len(snippets) >= 2
