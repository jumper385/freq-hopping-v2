import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from src.PlutoSDR import PlutoSDR
from src.bb.BaseBand import BaseBand
from src.bb.Constellation import Constellation

# --- Hardware setup (one-time) ---
sdr_rx = PlutoSDR(uri="ip:192.168.8.93", 
                  buffer_size=1024*3, 
                  center_freq=915_000_000)

sdr_tx = PlutoSDR(uri="usb:", 
                  tx_gain=0, 
                  center_freq=915_000_000)

bb = BaseBand(seq_len=512, 
              pre_len=32, 
              center_guard=32, 
              pilot_spacing=3)

const_N = 8
const = Constellation(const_N)

# --- pyqtgraph window ---
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

app = QtWidgets.QApplication(sys.argv)

# --- Main container ---
main_widget = QtWidgets.QWidget()
main_widget.setWindowTitle("Real-time SDR")
main_layout = QtWidgets.QVBoxLayout(main_widget)
main_layout.setContentsMargins(6, 6, 6, 6)
main_layout.setSpacing(4)

# --- Control bar ---
ctrl_layout = QtWidgets.QHBoxLayout()
ctrl_layout.addWidget(QtWidgets.QLabel("QAM modulation:"))
qam_combo = QtWidgets.QComboBox()
_qam_options = [4, 6, 8]
for n in _qam_options:
    qam_combo.addItem(f"{2**n}-QAM  ({n} bits/sym)", userData=n)
qam_combo.setCurrentIndex(_qam_options.index(const_N))
ctrl_layout.addWidget(qam_combo)
ctrl_layout.addStretch()
main_layout.addLayout(ctrl_layout)

# --- Plots ---
win = pg.GraphicsLayoutWidget()
main_layout.addWidget(win)
main_widget.resize(1200, 720)
main_widget.show()

p1 = win.addPlot(title="Received RF Signal", colspan=2)
p1.setLabel('bottom', 'Sample')
curve_rf_real = p1.plot(pen=pg.mkPen('k', width=0.5))
curve_rf_imag = p1.plot(pen=pg.mkPen('r', width=0.5))

win.nextRow()

p2 = win.addPlot(title="Detected Symbols")
p2.setLabel('bottom', 'Tone')
p2.setYRange(-2.5, 2.5)
p2.enableAutoRange(axis='x')
curve_sym_real = p2.plot(pen=pg.mkPen('k', width=0.5))
curve_sym_imag = p2.plot(pen=pg.mkPen('b', width=0.5))

p3 = win.addPlot(title="Constellation")
p3.setLabel('bottom', 'I')
p3.setLabel('left', 'Q')
p3.setAspectLocked(True)
p3.setXRange(-1.5, 1.5)
p3.setYRange(-1.5, 1.5)
p3.disableAutoRange()
scatter = pg.ScatterPlotItem(size=5, 
                             pen=None, 
                             brush=pg.mkBrush(30, 100, 200, 140))
p3.addItem(scatter)


# --- QAM change callback ---
def on_qam_changed():
    global const
    n = qam_combo.currentData()
    const = Constellation(n)

qam_combo.currentIndexChanged.connect(on_qam_changed)

# --- FPS tracking ---
_fps_last_time = time.perf_counter()
_fps_alpha = 0.1  # exponential moving average smoothing
_fps_ema = 0.0


# --- Real-time update callback ---
def update():
    global _fps_last_time, _fps_ema

    now = time.perf_counter()
    dt = now - _fps_last_time
    _fps_last_time = now
    if dt > 0:
        _fps_ema = _fps_alpha * (1.0 / dt) + (1 - _fps_alpha) * _fps_ema
    main_widget.setWindowTitle(f"Real-time SDR  |  {_fps_ema:.1f} frames/s")

    payload = f"The quick brown fox jumped over the lazy hare"
    payload = payload.encode("utf8")
    sig_x = const.map(payload, bb.max_data_len * const.N)
    sig = bb.gen_tx(sig_x)
    sdr_tx.transmit(np.tile(sig * 2**10, 25))

    sig_rx = sdr_rx.receive(flush=0)

    curve_rf_real.setData(sig_rx.real)
    curve_rf_imag.setData(sig_rx.imag)

    det = bb.det_rx(sig_rx)
    if len(det) > 0:
        symbol = np.mean(det, axis=0)
        curve_sym_real.setData(symbol.real)
        curve_sym_imag.setData(symbol.imag)

        scatter.setData(symbol.real, symbol.imag)

        yhat = const.demap(symbol)

        if yhat is not None:
            print(yhat.decode("utf8", errors="ignore"))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)  # 0 ms → fire as fast as the event loop allows

try:
    sys.exit(app.exec())
finally:
    sdr_tx.stop_transmit()
