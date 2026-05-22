import numpy as np
import matplotlib.pyplot as plt

from src.PlutoSDR import PlutoSDR
from src.bb.BaseBand import BaseBand


sdr_rx = PlutoSDR(uri="ip:192.168.8.94", buffer_size=1024*36)
sdr_tx = PlutoSDR(uri="ip:192.168.8.93", tx_gain=0)

bb = BaseBand(seq_len=1024, pre_len=32, _enable_cfo=False)
sig = bb.gen_tx()

sdr_tx.transmit(sig * 2**12)
sig_rx = sdr_rx.receive(3)
t = np.arange(0, len(sig_rx))
# sig_rx = sig_rx * np.exp(-2j * np.pi * t * 2.08 / 1024)

fig, ax = plt.subplots(4, sharex=False)

# plot received signal
ax[0].plot(sig_rx.real, c='black', lw=0.5)
ax[0].plot(sig_rx.imag, c='red', lw=0.5)

# plot detections
det = bb.det_rx(sig_rx)
for symbol in det:
    ax[1].plot(symbol.real, c='black', lw=0.5)
    ax[1].plot(symbol.imag, c='blue', lw=0.5)

ax[2].set_xlim(-10, 600)

corr = np.correlate(sig_rx, bb.preamble)
corr_mag = np.abs(corr)
corr_thresh = bb._get_threshold(np.abs(corr_mag))
print(corr_thresh)
rising_edges = bb._get_preambles(corr_mag, corr_thresh)

ax[3].plot(corr_mag.real, lw=0.5, c='black')
ax[3].plot(corr_mag.imag, lw=0.5, c='red')
ax[3].axhline(corr_thresh, lw=0.5, c='blue')

ax[3].set_title("Complex Correlation Plot of Recived Signal and Preamble")
ax[2].set_title("FFT of the Received Signal")
ax[1].set_title("Detected Snippets")
ax[0].set_title("Received RF Signal")

fig.tight_layout()

plt.show()
