import zlib
import numpy as np


class Framer:
    """
    Handles byte-level packet framing and CRC verification.

    Packet layout:
        [2-byte big-endian payload length | payload | 4-byte CRC32]

    CRC is computed over the length prefix + payload.
    """

    HEADER_LEN = 2
    CRC_LEN = 4
    OVERHEAD = HEADER_LEN + CRC_LEN

    def pack(self, payload: bytes) -> bytes:
        if len(payload) > 0xFFFF:
            raise ValueError("Payload too large for 2-byte length field")
        length_prefix = len(payload).to_bytes(self.HEADER_LEN, "big")
        protected = length_prefix + payload
        crc = (zlib.crc32(protected) & 0xFFFFFFFF).to_bytes(self.CRC_LEN, "big")
        return protected + crc

    def unpack(self, raw: bytes) -> bytes | None:
        if len(raw) < self.OVERHEAD:
            return None
        length = int.from_bytes(raw[:self.HEADER_LEN], "big")
        packet_len = self.HEADER_LEN + length + self.CRC_LEN
        if packet_len > len(raw):
            return None
        protected = raw[:self.HEADER_LEN + length]
        payload = raw[self.HEADER_LEN:self.HEADER_LEN + length]
        raw_crc = int.from_bytes(raw[self.HEADER_LEN + length:packet_len], "big")
        calc_crc = zlib.crc32(protected) & 0xFFFFFFFF
        if raw_crc != calc_crc:
            return None
        return payload

    def bits_needed(self, payload: bytes) -> int:
        return (len(payload) + self.OVERHEAD) * 8
