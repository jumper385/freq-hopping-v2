"""
Tests for src/bb/framing.py
"""
import pytest
import numpy as np
from src.bb.framing import Framer


@pytest.fixture
def framer():
    return Framer()


class TestFramerPack:
    def test_roundtrip_simple(self, framer):
        payload = b"hello world"
        packed = framer.pack(payload)
        assert framer.unpack(packed) == payload

    def test_roundtrip_empty(self, framer):
        packed = framer.pack(b"")
        assert framer.unpack(packed) == b""

    def test_roundtrip_max_size(self, framer):
        payload = bytes(range(256)) * 128  # 32 768 bytes
        packed = framer.pack(payload)
        assert framer.unpack(packed) == payload

    def test_pack_overhead(self, framer):
        payload = b"abc"
        packed = framer.pack(payload)
        assert len(packed) == len(payload) + framer.OVERHEAD

    def test_payload_too_large_raises(self, framer):
        with pytest.raises(ValueError):
            framer.pack(b"x" * (0xFFFF + 1))


class TestFramerUnpack:
    def test_bit_flip_fails_crc(self, framer):
        payload = b"test payload"
        packed = bytearray(framer.pack(payload))
        packed[3] ^= 0xFF  # corrupt a payload byte
        assert framer.unpack(bytes(packed)) is None

    def test_truncated_returns_none(self, framer):
        packed = framer.pack(b"hello")
        assert framer.unpack(packed[:2]) is None

    def test_empty_bytes_returns_none(self, framer):
        assert framer.unpack(b"") is None

    def test_extra_trailing_bytes_ok(self, framer):
        payload = b"data"
        packed = framer.pack(payload) + b"\x00\x00\x00\x00"
        assert framer.unpack(packed) == payload


class TestFramerBitsNeeded:
    def test_bits_needed(self, framer):
        payload = b"hi"
        assert framer.bits_needed(payload) == (len(payload) + framer.OVERHEAD) * 8
