import numpy as np
import matplotlib.pyplot as plt

from src.PlutoSDR import PlutoSDR
from src.bb.BaseBand import BaseBand


sdr_rx = PlutoSDR(uri="ip:192.168.8.94", buffer_size=1024*36)
sdr_tx = PlutoSDR(uri="ip:192.168.8.93", tx_gain=0)

bb = BaseBand(seq_len=600, pre_len=64, _enable_cfo=True)
sig = bb.gen_tx()

sdr_tx.transmit(sig * 2**12)
sig_rx = sdr_rx.receive(3)
t = np.arange(0, len(sig_rx))
# sig_rx = sig_rx * np.exp(-2j * np.pi * t * 2.08 / 1024)

fig, ax = plt.subplots(5, sharex=False)

# plot received signal
ax[0].plot(sig_rx.real, c='black', lw=0.5)
ax[0].plot(sig_rx.imag, c='red', lw=0.5)

# plot detections
det = bb.det_rx(sig_rx)
for symbol in det:
    ax[1].plot(symbol.real, c='black', lw=0.5)
    ax[1].plot(symbol.imag, c='black', lw=0.5)

    fft = np.fft.fft(symbol)
    # freqs = np.fft.fftfreq(len(symbol), d=1/len(symbol))
    ax[2].plot(fft.real, c='red', lw=0.5)
    ax[2].plot(fft.imag, c='blue', lw=0.5)

ax[2].set_xlim(-10, 220)

corr = np.correlate(sig_rx, bb.preamble)
corr_mag = np.abs(corr)
corr_thresh = bb._get_threshold(np.abs(corr_mag))
print(corr_thresh)
rising_edges = bb._get_preambles(corr_mag, corr_thresh)

ax[3].plot(corr_mag.real, lw=0.5, c='black')
ax[3].plot(corr_mag.imag, lw=0.5, c='red')
ax[3].axhline(corr_thresh, lw=0.5, c='blue')

# plot the polarity in for detected rising_edges
corr_angle = np.angle(corr)[rising_edges]
print(corr_angle)
print(rising_edges)
ax[4].set_title("Phase Angle vs Complex Preable Correlation")
ax[4].plot(corr.real, lw=0.5, c='black', label="Correlation (REAL)")
ax[4].plot(corr.imag, lw=0.5, c='red', label="Correlation (IMAG)")
ax[4].scatter(rising_edges, corr_angle, marker='x', c='blue', s=10, label="Phase Ambiguity (rad)")

ax[3].set_title("Complex Correlation Plot of Recived Signal and Preamble")
ax[2].set_title("FFT of the Received Signal")
ax[1].set_title("Detected Snippets")
ax[0].set_title("Received RF Signal")

fig.tight_layout()

plt.show()
