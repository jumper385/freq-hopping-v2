import torch
import numpy as np
from sionna.phy.mapping import Mapper, Demapper


class Constellation:
    def __init__(self, N):
        self.N = N
        self.mapper = Mapper("qam", N)
        self.demapper = Demapper("app", "qam", 4, hard_out=True)

        self.dbg_data = None

    def map(self, seq_len):
        """
        takes bits as 0bXXXX and maps it to an array of complex symbols
        """
        bits = torch.randint(0, 2, (1, seq_len), dtype=torch.float32)
        bits = bits.reshape(-1)
        symbols = self.mapper(bits)

        self.dbg_data = bits

        return symbols.numpy()

    def demap(self, y):
        y_torch = torch.tensor(y)
        llr = self.demapper(y_torch, no=0.01)

        if self.dbg_data != None:
            error = torch.sum(self.dbg_data != llr).item()
            print(f"Bit Errors: {error}")

        return llr.numpy()
