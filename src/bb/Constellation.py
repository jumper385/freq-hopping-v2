import torch
import numpy as np

from sionna.phy.mapping import Mapper, Demapper


class Constellation:
    def __init__(self, N):
        self.N = N
        self.mapper = Mapper("qam", N)
        self.demapper = Demapper("app", "qam", 4, hard_out=True)

        self.dbg_data = None
        self.total_symbols = 0
        self.error_count = 0

    def map(self, payload, max_bit_len):
        """
        takes bits as 0bXXXX and maps it to an array of complex symbols
        """
        bits = torch.randint(0, 2, (1, max_bit_len), dtype=torch.float32)

        # encode
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        pad_len = max_bit_len - len(bits)
        pad_zero = np.zeros(pad_len)
        bits = np.concat([bits, pad_zero])

        bits = bits.reshape(-1)
        bits = torch.tensor(bits)
        symbols = self.mapper(bits)

        self.dbg_data = bits

        return symbols.numpy()

    def demap(self, y):
        y_torch = torch.tensor(y)
        llr = self.demapper(y_torch, no=0.01)

        if self.dbg_data != None:
            self.total_symbols += 1
            error = torch.sum(self.dbg_data != llr).item()

            error_rate = self.error_count / self.total_symbols
            # print(f"Bit Errors: {error} / {len(self.dbg_data)} {error_rate*100:.2f}%")

            if error > 0:
                self.error_count += 1
                return None
            else:
                bits = llr.int().tolist()
                payload = np.packbits(bits).tobytes()
                return payload

        return None
