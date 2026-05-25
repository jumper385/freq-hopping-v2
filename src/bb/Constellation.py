import zlib
import torch
import numpy as np
from datetime import datetime

from sionna.phy.mapping import Mapper, Demapper


class Constellation:
    def __init__(self, N):
        self.N = N
        self.mapper = Mapper("qam", N)
        self.demapper = Demapper("app", "qam", N, hard_out=True)

        self.dbg_data = None
        self.total_symbols = 0
        self.error_count = 0

    def map(self, payload: bytes, max_bit_len: int):
        """
        Packet structure:
            [2-byte payload length | payload | 4-byte crc32 | padding]

        CRC is computed over:
            [length | payload]
        """

        if len(payload) > 0xFFFF:
            raise ValueError("Payload too large for 2-byte length field")

        length_prefix = len(payload).to_bytes(2, "big")

        protected = length_prefix + payload
        crc = zlib.crc32(protected) & 0xFFFFFFFF

        packet = protected + crc.to_bytes(4, "big")

        required_bits = len(packet) * 8
        if required_bits > max_bit_len:
            raise ValueError(
                f"Packet too large: needs {required_bits} bits, frame has {max_bit_len}"
            )

        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))

        pad_len = max_bit_len - len(bits)
        pad_zero = np.zeros(pad_len, dtype=np.uint8)

        bits = np.concatenate([bits, pad_zero])
        bits = torch.tensor(bits, dtype=torch.float32)

        symbols = self.mapper(bits)

        self.dbg_data = bits

        return symbols.cpu().numpy()

    def demap(self, y):
        y_torch = torch.tensor(y)
        bits = self.demapper(y_torch, no=0.01)

        bits = bits.int().cpu().numpy().astype(np.uint8)
        raw = np.packbits(bits).tobytes()

        if len(raw) < 2 + 4:
            print("FAILED: frame too short")
            return None

        length = int.from_bytes(raw[:2], "big")

        packet_len = 2 + length + 4

        if packet_len > len(raw):
            # print(f"FAILED: invalid length {length}, raw frame only {len(raw)} bytes")
            return None

        protected = raw[:2 + length]
        payload = raw[2:2 + length]

        raw_crc = int.from_bytes(raw[2 + length:2 + length + 4], "big")
        calc_crc = zlib.crc32(protected) & 0xFFFFFFFF

        if raw_crc != calc_crc:
#            print(
#                f"FAILED: len={length}, raw_crc={raw_crc:#010x}, "
#                f"calc_crc={calc_crc:#010x}, time={datetime.now()}"
#            )
#
#            # Do not decode corrupted payload as UTF-8. It may not be valid text.
#            print(f"raw header bytes: {raw[:8].hex()}")
#            print(f"payload preview: {payload[:32].hex()}")
            return None

        return payload
