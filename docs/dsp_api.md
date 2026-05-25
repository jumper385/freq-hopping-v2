# DSP Building-Block API Reference

This document describes the contracts for every module under `src/bb/`.  
Read it to understand how each block works, how data flows between them,  
and how to modify or swap out individual stages.

---

## Table of Contents

1. [Quick-Start](#1-quick-start)
2. [Architecture Overview](#2-architecture-overview)
3. [Wire Format](#3-wire-format)
4. [OFDM Tone Layout](#4-ofdm-tone-layout)
5. [Block Reference](#5-block-reference)
   - [5.1 Framer](#51-framer--srcbbframingpy)
   - [5.2 QAMMapper](#52-qammapper--srcbbmappingpy)
   - [5.3 ToneLayout + OFDMModulator](#53-tonelayout--ofdmmodulator--srcbbofdmpy)
   - [5.4 PreambleSync](#54-preamblesync--srcbbsyncpy)
   - [5.5 PilotEqualizer](#55-pilotequalizer--srcbbequalizerpy)
   - [5.6 TxPipeline / RxPipeline / build_pipelines](#56-txpipeline--rxpipeline--build_pipelines--srcbbpipelinepy)
6. [Default Parameters](#6-default-parameters)
7. [Modifying the System](#7-modifying-the-system)
8. [Constraints and Gotchas](#8-constraints-and-gotchas)

---

## 1. Quick-Start

```python
from src.bb.pipeline import build_pipelines

tx, rx = build_pipelines()

# --- Transmit ---
frame = tx.encode(b"hello world")   # np.ndarray of int16-scale complex samples
sdr.transmit(frame)

# --- Receive ---
rx_samples = sdr.receive(n_samples)
msg = rx.decode_averaged(rx_samples)   # bytes | None
if msg:
    print(msg)
```

`build_pipelines()` is the only entry point you normally need. It creates all  
blocks with a shared `ToneLayout` and preamble so Tx and Rx are always consistent.

---

## 2. Architecture Overview

```
                         TRANSMIT PATH
  bytes
    │
    ▼
┌─────────┐   CRC-framed packet   ┌───────────┐   complex symbols   ┌───────────────┐
│  Framer │──────────────────────▶│ QAMMapper │────────────────────▶│OFDMModulator  │
└─────────┘       (bytes)         └───────────┘    (np.ndarray)     └───────────────┘
                                                                            │ freq-domain vector
                                                                            │ IFFT + CP + preamble
                                                                            ▼
                                                               SDR-ready samples (×tx_scale, ×tile)


                         RECEIVE PATH
  SDR samples
    │
    ▼
┌──────────────┐  time-domain snippets  ┌───────────────┐  active slice  ┌───────────────┐
│ PreambleSync │───────────────────────▶│OFDMModulator  │───────────────▶│ PilotEqualizer│
└──────────────┘  (per-frame arrays)    │  .demodulate()│                └───────────────┘
                                        └───────────────┘                        │ data tones
                                                                                 ▼
                                                                          ┌───────────┐
                                                                          │ QAMMapper │
                                                                          │  .demap() │
                                                                          └───────────┘
                                                                                 │ bit array
                                                                                 ▼
                                                                          ┌─────────┐
                                                                          │  Framer │
                                                                          │ .unpack()│
                                                                          └─────────┘
                                                                                 │
                                                                           bytes | None
```

### Shared state

`build_pipelines()` gives both pipelines the same `ToneLayout` instance and  
the same `preamble` array.  This means any structural change (tone count,  
guard sizes, pilot spacing) only needs to be made in one place.

---

## 3. Wire Format

### Packet (Framer output, used as payload for QAM mapping)

```
┌─────────────────────┬─────────────────────────┬──────────────────────────┐
│  Length (2 bytes BE)│  Payload (variable)      │  CRC32 (4 bytes BE)      │
└─────────────────────┴─────────────────────────┴──────────────────────────┘
```

- Length field: `uint16` big-endian, value = `len(payload)`
- CRC32: computed over `length_prefix + payload` (not over the CRC itself)
- Total overhead: **6 bytes** (`OVERHEAD = HEADER_LEN + CRC_LEN = 2 + 4`)
- Maximum payload: `(max_data_tones * qam_bits // 8) - 6` bytes

### Tx Frame (one tile, on the wire)

```
┌──────────┬──────────┬────────────────┬──────────────────────────┐
│ Preamble │ Preamble │  Cyclic Prefix │   OFDM symbol (n_tones)  │
│ (pre_len)│ (pre_len)│   (cp_len)     │      = seq_len // 2      │
└──────────┴──────────┴────────────────┴──────────────────────────┘
```

The full transmitted buffer is this structure scaled by `tx_scale` and  
repeated `tile` times end-to-end.

- **Preamble**: Zadoff-Chu sequence, root `n = pre_len - 1`, length `pre_len`.
  Two copies are sent so that the receiver can use phase difference between  
  them to estimate carrier-frequency offset (CFO).
- **Cyclic prefix**: the last `cp_len` samples of the OFDM symbol, prepended  
  to reduce ISI in multipath channels.
- **OFDM symbol**: `np.fft.ifft(freq_vector)`, normalised by `_power_match()`  
  before scaling.

---

## 4. OFDM Tone Layout

The system uses a **half-spectrum** convention: FFT size is `n_tones = seq_len // 2`.  
The frequency-domain vector is divided into zones:

```
Index 0                                                        n_tones-1
│◄── guard_pad ──►│◄─────────────── active ──────────────────►│◄── guard_pad ──►│
                  │ pilots | data | [center_guard] | data | pilots │
```

### ToneLayout attributes

| Attribute        | Type            | Description                                                         |
|------------------|-----------------|---------------------------------------------------------------------|
| `active_len`     | `int`           | `n_tones - 2 * guard_pad`                                           |
| `center_mask`    | `ndarray[bool]` | `True` for DC-guard bins (width `2 * center_guard`) — always zeroed |
| `pilot_mask`     | `ndarray[bool]` | Every `pilot_spacing`-th active bin, excluding `center_mask`        |
| `data_mask`      | `ndarray[bool]` | `~pilot_mask & ~center_mask` — the usable data tones                |
| `pilot_refs`     | `ndarray[cplx]` | Alternating `1+1j` / `-1-1j` per pilot position                     |
| `pilot_idx`      | `ndarray[int]`  | Integer indices of pilot positions within the active slice           |
| `max_data_tones` | `int`           | Total number of usable data tones per OFDM symbol                   |

All masks index into the **active slice only** (length `active_len`), not  
into the full FFT vector.

---

## 5. Block Reference

### 5.1 `Framer` — `src/bb/framing.py`

Byte-level packet framing with CRC integrity check.

```python
framer = Framer()
```

No constructor parameters.

#### Methods

| Method                          | Input         | Output         | Notes                                          |
|---------------------------------|---------------|----------------|------------------------------------------------|
| `pack(payload: bytes) → bytes`  | raw payload   | framed packet  | Prepends 2-byte length, appends 4-byte CRC32   |
| `unpack(raw: bytes) → bytes\|None` | framed packet | payload or None | Returns `None` on CRC failure or truncation |
| `bits_needed(payload: bytes) → int` | payload  | int            | Total bit-count of the framed form `×8`        |

#### Constants

| Constant     | Value | Meaning                        |
|--------------|-------|--------------------------------|
| `HEADER_LEN` | 2     | Length-prefix bytes            |
| `CRC_LEN`    | 4     | CRC32 bytes                    |
| `OVERHEAD`   | 6     | `HEADER_LEN + CRC_LEN`         |

---

### 5.2 `QAMMapper` — `src/bb/mapping.py`

Wraps Sionna's GPU-aware QAM Mapper/Demapper.

> [!WARNING]
> Will remove this with a custom QAM mapper oneday

```python
mapper = QAMMapper(N=4)   # 16-QAM
```

| Parameter | Type | Description                      |
|-----------|------|----------------------------------|
| `N`       | int  | Bits per symbol (4→16-QAM, 6→64-QAM) |

#### Methods

| Method                               | Input                                  | Output               | Notes                                  |
|--------------------------------------|----------------------------------------|----------------------|----------------------------------------|
| `map(bits: np.ndarray) → np.ndarray` | `uint8` flat bit array, length ÷ N    | complex128 symbols   | Length must be divisible by `N`        |
| `demap(symbols: np.ndarray) → np.ndarray` | complex symbols                  | `uint8` bit array    | Hard decision; internally uses `no=0.01` |

> **CUDA note**: Sionna may move tensors to GPU automatically.  
> Both methods call `.cpu()` before `.numpy()` — safe on CPU-only machines too.

---

### 5.3 `ToneLayout` + `OFDMModulator` — `src/bb/ofdm.py`

#### `ToneLayout(n_tones, guard_pad, pilot_spacing, center_guard)`

Compute-once object. Pass the same instance to `OFDMModulator`,  
`PilotEqualizer`, and (via the pipeline) everything else.

| Parameter       | Type | Description                                               |
|-----------------|------|-----------------------------------------------------------|
| `n_tones`       | int  | FFT size = `seq_len // 2`                                 |
| `guard_pad`     | int  | Zero-padded tones at each spectrum edge                   |
| `pilot_spacing` | int  | One pilot every N active tones                            |
| `center_guard`  | int  | Half-width of DC null region (tones zeroed around DC bin) |

#### `OFDMModulator(layout: ToneLayout)`

| Method                                              | Input                                   | Output                     | Notes                                     |
|-----------------------------------------------------|-----------------------------------------|----------------------------|-------------------------------------------|
| `modulate(data_symbols: np.ndarray) → np.ndarray`  | complex, length ≤ `max_data_tones`      | freq-domain vector, n_tones | Guard zeroed, pilots inserted. No IFFT.  |
| `demodulate(freq_vector: np.ndarray) → np.ndarray` | complex, length = `n_tones`             | active slice, `active_len` | Strips guard; does NOT equalize.          |

> The pipeline applies `np.fft.ifft` / `np.fft.fft` around these calls.  
> Modulator and equalizer only see the frequency domain.

---

### 5.4 `PreambleSync` — `src/bb/sync.py`

Detects OFDM frames in a raw RX sample stream and optionally corrects CFO.

```python
sync = PreambleSync(preamble, cp_len=32, seq_len=1024, enable_cfo=True)
```

| Parameter    | Type           | Default                    | Description                                    |
|--------------|----------------|----------------------------|------------------------------------------------|
| `preamble`   | `np.ndarray`   | —                          | Known ZC sequence (complex), length `pre_len`  |
| `cp_len`     | `int`          | —                          | Cyclic prefix length in samples                |
| `seq_len`    | `int`          | —                          | OFDM symbol FFT size (full, not half)          |
| `enable_cfo` | `bool`         | `True`                     | Apply CFO correction to each extracted snippet |
| `margin`     | `int \| None`  | `max(1, pre_len × 0.1)`    | Tolerance on preamble-pair spacing             |

#### Methods

| Method                                       | Input           | Output                            |
|----------------------------------------------|-----------------|-----------------------------------|
| `find_frames(rx: np.ndarray) → list[np.ndarray]` | complex samples | List of time-domain snippets, each shape `(n_tones,)` |
| `correlation_magnitude(rx: np.ndarray) → np.ndarray` | complex samples | Raw `|cross-correlation|` for debugging |

#### Algorithm

1. `corr_mag = |np.correlate(rx, preamble, mode='valid')|`
2. Adaptive threshold: `(max(corr_mag) - min(corr_mag)) / 2`
3. Rising-edge indices from `np.diff(corr_mag > threshold)`
4. Validate consecutive edge pairs: spacing ≈ `pre_len` (within ± `margin`)
5. For each valid pair `(edge_i, edge_i+1)`:
   - Snippet start = `edge_i+1 + 1 + pre_len + cp_len`
   - Snippet length = `n_tones = seq_len // 2`
6. If `enable_cfo`: estimate CFO from phase rotation between the two preamble copies,  
   then rotate snippet by `exp(-2πj·t·(-cfo)/seq_len)`

**CFO formula**:

$$\text{CFO} = \frac{\angle(\mathbf{p}_1 \cdot \mathbf{p}_2^*)}{2\pi \cdot \frac{pre\_len}{seq\_len}}$$

> **Constraint**: Requires at least **2 preamble peaks** (3 edges for a single frame).  
> A single isolated frame with very short leading silence may not be detected  
> reliably due to the adaptive threshold using the global signal range.  
> Use `tile ≥ 2` in practice, or add leading silence ≥ `2 × pre_len`.

---

### 5.5 `PilotEqualizer` — `src/bb/equalizer.py`

Single-tap frequency-domain channel equalizer using known pilot tones.

```python
eq = PilotEqualizer(layout)
```

#### Method

```python
equalize(active_symbols: np.ndarray) → np.ndarray
```

| Argument         | Shape                    | Description                                        |
|------------------|--------------------------|----------------------------------------------------|
| `active_symbols` | `(n_symbols, active_len)`| Freq-domain active slice (guard bins removed)      |
| **Returns**      | `(n_symbols, max_data_tones)` | Equalized data tones (pilots + DC-guard removed) |

#### Algorithm (per symbol)

1. Extract received pilots: `active_symbols[:, pilot_mask]`
2. Divide by `pilot_refs` → channel estimate at pilot positions `H_k`
3. Linearly interpolate `re(H)` and `im(H)` across all active tone indices
4. Divide all active tones by interpolated `H`
5. Return only the `data_mask` tones

---

### 5.6 `TxPipeline` / `RxPipeline` / `build_pipelines` — `src/bb/pipeline.py`

#### `build_pipelines(**kwargs) → tuple[TxPipeline, RxPipeline]`

Factory that creates all blocks with shared structural objects.

```python
tx, rx = build_pipelines(
    seq_len=1024,
    pre_len=32,
    cp_len=32,
    guard_pad=16,
    pilot_spacing=8,
    center_guard=32,
    qam_bits=4,
    enable_cfo=True,
    tx_scale=2**14,
    tile=24,
)
```

All parameters are keyword-only and have defaults (see [§6](#6-default-parameters)).

#### `TxPipeline`

| Method / Property         | Description                                                            |
|---------------------------|------------------------------------------------------------------------|
| `encode(payload: bytes) → np.ndarray` | Full pipeline: framing → QAM map → OFDM mod → IFFT → preamble + CP → tile + scale |
| `max_payload_bytes`       | Maximum payload size in bytes for one frame                           |

Internal stages of `encode`:
1. `Framer.pack` → packet bytes
2. `np.unpackbits` → bit array (zero-padded to `max_data_tones × N`)
3. `QAMMapper.map` → complex data symbols
4. `OFDMModulator.modulate` → frequency-domain vector
5. `np.fft.ifft` → time-domain symbol
6. `_power_match(preamble, time_domain)` → power normalisation
7. Prepend `[preamble | preamble | CP]`
8. Multiply by `tx_scale`, repeat `tile` times

#### `RxPipeline`

| Method                                    | Description                                                |
|-------------------------------------------|------------------------------------------------------------|
| `decode(rx: np.ndarray) → list[bytes\|None]` | One result per detected frame; `None` = CRC failure     |
| `decode_averaged(rx: np.ndarray) → bytes\|None` | Averages all detected symbol copies before demapping  |

`decode_averaged` is preferred when the frame is tiled many times — averaging  
over `tile` copies gives a significant SNR improvement before hard QAM decisions.

Internal stages of `decode` / `decode_averaged`:
1. `PreambleSync.find_frames` → list of time-domain snippets
2. `np.fft.fft` on each snippet
3. `OFDMModulator.demodulate` → active slice per frame
4. `PilotEqualizer.equalize` → data tones
5. *(averaged only)* `np.mean(axis=0)` across all frames
6. `QAMMapper.demap` → bit array
7. `np.packbits` → bytes
8. `Framer.unpack` → payload or `None`

---

## 6. Default Parameters

| Parameter       | Default    | Used in              | Description                              |
|-----------------|------------|----------------------|------------------------------------------|
| `seq_len`       | `1024`     | everywhere           | OFDM symbol FFT size (full)              |
| `pre_len`       | `32`       | sync                 | Preamble / Zadoff-Chu length             |
| `cp_len`        | `32`       | tx, sync             | Cyclic prefix length                     |
| `guard_pad`     | `16`       | ToneLayout           | Edge-guard zero tones                    |
| `pilot_spacing` | `8`        | ToneLayout           | One pilot per this many active tones     |
| `center_guard`  | `32`       | ToneLayout           | DC-guard half-width in tones             |
| `qam_bits`      | `4`        | QAMMapper            | Bits per symbol (4 = 16-QAM)             |
| `enable_cfo`    | `True`     | PreambleSync         | Enable CFO correction                    |
| `tx_scale`      | `2**14`    | TxPipeline           | Integer scale for SDR DAC range          |
| `tile`          | `24`       | TxPipeline           | Number of frame repetitions per buffer   |

**Derived quantities** (with defaults):

| Derived                | Value                        | Formula                                     |
|------------------------|------------------------------|---------------------------------------------|
| `n_tones`              | `512`                        | `seq_len // 2`                              |
| `active_len`           | `480`                        | `n_tones - 2 * guard_pad`                   |
| `max_data_tones`       | ≈ `400` (depends on pilots)  | `active_len - n_pilots - 2*center_guard`    |
| `max_payload_bytes`    | ≈ `194`                      | `max_data_tones * qam_bits // 8 - OVERHEAD` |

---

## 7. Modifying the System

### Change modulation order

```python
tx, rx = build_pipelines(qam_bits=6)   # 64-QAM
```

Higher `qam_bits` → more bits per symbol → larger payload capacity, but  
less noise resilience. Always use `build_pipelines` so Tx and Rx stay matched.

### Change FFT / symbol size

```python
tx, rx = build_pipelines(seq_len=512)  # half the default
```

Reduces latency and memory but halves `max_data_tones`.

### Swap pilot spacing (channel coherence)

Denser pilots (`pilot_spacing=4`) improve equalization accuracy in  
fast-varying channels at the cost of fewer data tones.

```python
tx, rx = build_pipelines(pilot_spacing=4)
```

### Disable CFO correction

Useful when the Tx and Rx share a clock reference:

```python
tx, rx = build_pipelines(enable_cfo=False)
```

### Use individual blocks without the pipeline

All blocks are independently importable:

```python
from src.bb import Framer, QAMMapper, ToneLayout, OFDMModulator, PreambleSync, PilotEqualizer
```

### Replace the preamble sequence

`PreambleSync` accepts any complex preamble. The default is Zadoff-Chu from  
`src/utils/preambles.py`. To use a custom sequence:

```python
from src.bb.sync import PreambleSync
import numpy as np

my_preamble = np.random.randn(64) + 1j * np.random.randn(64)
sync = PreambleSync(my_preamble, cp_len=32, seq_len=1024)
```

Pass the same array to `TxPipeline(preamble=my_preamble, ...)`.

### Add a new DSP block

1. Create `src/bb/myblock.py` with a class that takes `ToneLayout` if it  
   needs tone geometry.
2. Export it from `src/bb/__init__.py`.
3. Call it inside `TxPipeline.encode` or `RxPipeline.decode` at the  
   appropriate stage, or wire it as a standalone step outside the pipeline.

---

## 8. Constraints and Gotchas

| Constraint | Detail |
|---|---|
| **Bit count divisible by N** | `QAMMapper.map` requires `len(bits) % N == 0`. The pipeline always pads to `max_data_tones × N` before calling `map`. If you call `map` directly, pad first. |
| **tile ≥ 2 for reliable sync** | `PreambleSync` uses `np.diff` to find rising edges; an event at sample index 0 is invisible. Always transmit with at least 2 leading silence samples or `tile ≥ 2`. |
| **CUDA safety** | Sionna's Mapper/Demapper may run on GPU. Both `QAMMapper` methods call `.cpu()` before `.numpy()`. Never call `.numpy()` directly on a Sionna output without `.cpu()` first. |
| **Single-frame detection sensitivity** | Adaptive threshold `(max-min)/2` is sensitive to how much of the buffer is occupied. A single frame in a large buffer of zeros may not be detected reliably on all platforms. Use `tile ≥ 2` or prepend at least `2 × pre_len` samples of silence. |
| **Payload size limit** | `max_payload_bytes` depends on `qam_bits`, `seq_len`, `guard_pad`, `pilot_spacing`, and `center_guard`. Check `tx.max_payload_bytes` at runtime if you change any of these. |
| **`_power_match` is internal** | It normalises the OFDM symbol to 3× the preamble RMS power then clips to unit max. It is not part of the public API and may change. |
| **Interpolation at spectrum edges** | `scipy.interp1d` with `fill_value='extrapolate'` is used for pilot-based H estimation. Accuracy degrades at the edges of the active band where there are no pilots beyond the boundary. Widen `center_guard` or add guard pilots if edge tones are unreliable. |
