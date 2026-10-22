"""
Recording access. One place to adapt when a file format changes.

load(rx) returns a memory-mapped (n, n_ch) array, the sample rate, and the
receiver's clock offset guess (receiver_time = buoy_time + offset), so the
2 GB USBL file is never read whole. scipy reads the buoy PCM16 files and the
USBL WAVE_FORMAT_EXTENSIBLE float32 file (it has no .wav extension; scipy
does not care).
"""
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.io import wavfile


@dataclass
class Recording:
    name: str
    fs: int
    data: np.ndarray          # (n, n_ch), memory-mapped, native dtype
    offset: float             # receiver_time = buoy_time + offset [s]

    @property
    def n_ch(self):
        return self.data.shape[1]

    @property
    def duration(self):
        return self.data.shape[0] / self.fs

    def segment(self, t0, t1, ch=None, clock="buoy"):
        """float64 samples between t0 and t1 (buoy time by default, or the
        receiver's own clock with clock="rx"). Returns (x, t_first_sample)."""
        off = self.offset if clock == "buoy" else 0.0
        i0 = max(int(np.floor((t0 + off) * self.fs)), 0)
        i1 = min(int(np.ceil((t1 + off) * self.fs)), self.data.shape[0])
        x = self.data[i0:i1] if ch is None else self.data[i0:i1, ch]
        x = np.asarray(x, dtype=np.float64)
        if self.data.dtype == np.int16:
            x /= 32768.0
        return x, i0 / self.fs - off


def load(rx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        fs, x = wavfile.read(rx.file, mmap=True)
    if x.ndim == 1:
        x = x[:, None]
    return Recording(rx.name, fs, x, rx.clock_offset_guess)
