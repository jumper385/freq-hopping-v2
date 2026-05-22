import numpy as np
import matplotlib.pyplot as plt

from src.PlutoSDR import PlutoSDR
from src.bb.BaseBand import BaseBand


sdr_rx = PlutoSDR(uri="usb:0.1.5", buffer_size=1024*32)
sdr_tx = PlutoSDR(uri="ip:192.168.8.93", tx_gain=0)

bb = BaseBand(seq_len=512, pre_len=32, _enable_cfo=False)

sig_t = np.linspace(0, 1, 200)
sig_x = np.sin(2 * np.pi * 20 * sig_t)
sig_x = sig_x + sig_x * 1j

sig = bb.gen_tx(sig_x)

sdr_tx.transmit(sig * 2**14)
sig_rx = sdr_rx.receive(3)
t = np.arange(0, len(sig_rx))
# sig_rx = sig_rx * np.exp(-2j * np.pi * t * 2.08 / 1024)

fig, ax = plt.subplots(2, sharex=False)

# plot received signal
ax[0].plot(sig_rx.real, c='black', lw=0.5)
ax[0].plot(sig_rx.imag, c='red', lw=0.5)

# plot detections
det = bb.det_rx(sig_rx)
symbol = np.mean(det, axis=0)
ax[1].plot(symbol.real, c='black', lw=0.5)
ax[1].plot(symbol.imag, c='blue', lw=0.5)

corr = np.correlate(sig_rx, bb.preamble)
corr_mag = np.abs(corr)
corr_thresh = bb._get_threshold(np.abs(corr_mag))
rising_edges = bb._get_preambles(corr_mag, corr_thresh)

ax[1].set_title("Detected Snippets")
ax[0].set_title("Received RF Signal")

fig.tight_layout()

plt.show()
