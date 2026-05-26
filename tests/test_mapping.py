"""
Tests for src/bb/mapping.py
"""
import pytest
import numpy as np
from src.bb.mapping import QAMMapper


@pytest.fixture(params=[4, 6])
def mapper(request):
    return QAMMapper(request.param)


class TestQAMMapper:
    def test_map_returns_complex(self, mapper):
        n_bits = mapper.N * 16
        bits = np.random.randint(0, 2, n_bits).astype(np.uint8)
        symbols = mapper.map(bits)
        assert np.iscomplexobj(symbols)

    def test_map_output_length(self, mapper):
        n_bits = mapper.N * 16
        bits = np.zeros(n_bits, dtype=np.uint8)
        symbols = mapper.map(bits)
        assert len(symbols) == n_bits // mapper.N

    def test_demap_returns_uint8(self, mapper):
        bits = np.zeros(mapper.N * 16, dtype=np.uint8)
        symbols = mapper.map(bits)
        recovered = mapper.demap(symbols)
        assert recovered.dtype == np.uint8

    def test_demap_length_matches_input(self, mapper):
        bits = np.zeros(mapper.N * 16, dtype=np.uint8)
        symbols = mapper.map(bits)
        recovered = mapper.demap(symbols)
        assert len(recovered) == len(bits)

    def test_perfect_channel_roundtrip(self, mapper):
        """Noiseless roundtrip should recover bits exactly."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, mapper.N * 32).astype(np.uint8)
        symbols = mapper.map(bits)
        recovered = mapper.demap(symbols)
        assert np.array_equal(bits, recovered)

    def test_low_noise_roundtrip(self, mapper):
        """Very low noise should still recover bits correctly."""
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, mapper.N * 32).astype(np.uint8)
        symbols = mapper.map(bits)
        noisy = symbols + 0.01 * (rng.standard_normal(len(symbols)) +
                                  1j * rng.standard_normal(len(symbols)))
        recovered = mapper.demap(noisy)
        assert np.array_equal(bits, recovered)
