import torch
import pytest
import sionna.phy
import matplotlib.pyplot as plt

from src.bb.Constellation import Constellation

def test_constellation():

    mapper = sionna.phy.mapping.Mapper("qam", 4)
    demapper = sionna.phy.mapping.Demapper("app", "qam", 4, hard_out=True)

    bits = torch.randint(0, 2, (1, 100), dtype=torch.float32).reshape(-1)
    symbols = mapper(bits)

    # Add noise
    noise = 0.1 * torch.randn_like(symbols)
    y = symbols + noise

    # Compute LLRs
    llr = demapper(y, no=0.01)

    # bits_hat = (llr > 0).float()

    print(y)
    print(bits)
    print(llr)
    # print(bits_hat)
    # torch.Size([10, 100])

    print("bit errors:", torch.sum(bits != llr).item())
