import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from src.PlutoSDR import PlutoSDR
from src.bb.BaseBand import BaseBand
from src.bb.Constellation import Constellation

# --- Hardware setup (one-time) ---
sdr_rx = PlutoSDR(uri="usb:1.2.5", buffer_size=1024*8, center_freq=915_000_000)
sdr_tx = PlutoSDR(uri="ip:192.168.8.93", tx_gain=0, center_freq=915_000_000)

bb = BaseBand(seq_len=1024, pre_len=32, pilot_spacing=3)
const_N = 4
const = Constellation(const_N)

# --- pyqtgraph window ---
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

app = QtWidgets.QApplication(sys.argv)

win = pg.GraphicsLayoutWidget(title="Real-time SDR")
win.resize(1200, 700)
win.show()

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
scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(30, 100, 200, 140))
p3.addItem(scatter)


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
    win.setWindowTitle(f"Real-time SDR  |  {_fps_ema:.1f} frames/s")

    sig_x = const.map(bb.max_data_len * const_N)
    sig = bb.gen_tx(sig_x)
    sdr_tx.transmit(sig * 2**10)

    sig_rx = sdr_rx.receive(flush=3)

    curve_rf_real.setData(sig_rx.real)
    curve_rf_imag.setData(sig_rx.imag)

    det = bb.det_rx(sig_rx)
    if len(det) > 0:
        symbol = np.mean(det, axis=0)
        curve_sym_real.setData(symbol.real)
        curve_sym_imag.setData(symbol.imag)

        scatter.setData(symbol.real, symbol.imag)

        yhat = const.demap(symbol)
        print(yhat)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)  # 0 ms → fire as fast as the event loop allows

try:
    sys.exit(app.exec())
finally:
    sdr_tx.stop_transmit()
