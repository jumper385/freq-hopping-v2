"""
Tests for src/bb/pipeline.py  (TxPipeline, RxPipeline, build_pipelines)

These are end-to-end loopback tests – no hardware required.
The Tx frame is injected directly into the Rx pipeline through an optional
simulated channel (AWGN, flat fading).
"""
import pytest
import numpy as np
from src.bb.pipeline import build_pipelines, TxPipeline, RxPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipelines():
    return build_pipelines(
        seq_len=512,
        pre_len=32,
        cp_len=32,
        guard_pad=16,
        pilot_spacing=8,
        center_guard=32,
        qam_bits=4,
        enable_cfo=False,
        tx_scale=1.0,
        tile=3,  # tile>=2 ensures the first preamble always has a detectable rising edge
    )


@pytest.fixture
def tx(pipelines):
    return pipelines[0]


@pytest.fixture
def rx(pipelines):
    return pipelines[1]


# ---------------------------------------------------------------------------
# TxPipeline
# ---------------------------------------------------------------------------

class TestTxPipeline:
    def test_encode_returns_complex(self, tx):
        frame = tx.encode(b"hello")
        assert np.iscomplexobj(frame)

    def test_encode_nonzero(self, tx):
        frame = tx.encode(b"test")
        assert np.any(frame != 0)

    def test_max_payload_positive(self, tx):
        assert tx.max_payload_bytes > 0

    def test_payload_too_large_raises(self, tx):
        with pytest.raises(ValueError):
            tx.encode(b"x" * (tx.max_payload_bytes + 1))


# ---------------------------------------------------------------------------
# RxPipeline – clean loopback
# ---------------------------------------------------------------------------

class TestRxPipelineLoopback:
    def _loopback(self, tx, rx, payload, channel=None):
        """Encode, optionally apply a channel, then decode."""
        frame = tx.encode(payload)
        if channel is not None:
            frame = channel(frame)
        return rx.decode(frame)

    def test_decode_recovers_payload(self, tx, rx):
        payload = b"hello world"
        results = self._loopback(tx, rx, payload)
        assert any(r == payload for r in results), f"Got: {results}"

    def test_decode_returns_list(self, tx, rx):
        results = self._loopback(tx, rx, b"ping")
        assert isinstance(results, list)

    def test_decode_averaged_recovers_payload(self, tx, rx):
        payload = b"averaged test"
        tx_multi, rx_multi = build_pipelines(
            seq_len=512, pre_len=32, cp_len=32,
            guard_pad=16, pilot_spacing=8, center_guard=32,
            qam_bits=4, enable_cfo=False, tx_scale=1.0, tile=3
        )
        frame = tx_multi.encode(payload)
        result = rx_multi.decode_averaged(frame)
        assert result == payload

    def test_no_signal_returns_empty(self, rx):
        silence = np.zeros(2048, dtype=complex)
        assert rx.decode(silence) == []

    def test_binary_payload_roundtrip(self, tx, rx):
        payload = bytes(range(32))
        results = self._loopback(tx, rx, payload)
        assert any(r == payload for r in results)

    def test_flat_fading_channel(self, tx, rx):
        """Pilot equalization should handle a flat complex channel."""
        payload = b"fading test"
        h = 0.7 + 0.5j

        def flat_channel(x):
            return x * h

        results = self._loopback(tx, rx, payload, channel=flat_channel)
        assert any(r == payload for r in results), f"Got: {results}"


# ---------------------------------------------------------------------------
# build_pipelines factory
# ---------------------------------------------------------------------------

class TestBuildPipelines:
    def test_returns_tx_rx_pair(self):
        tx, rx = build_pipelines()
        assert isinstance(tx, TxPipeline)
        assert isinstance(rx, RxPipeline)

    def test_shared_layout(self):
        tx, rx = build_pipelines()
        assert tx.modulator.layout is rx.modulator.layout
