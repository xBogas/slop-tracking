import sys; sys.path.insert(0, "processing")
import numpy as np, config, recordings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter1d
cfg = config.load(); fs = 96000
R = {n: recordings.load(cfg.receiver(n)) for n in ["Rx902", "Rx01", "USBL"]}
def usbl_off(t): return 102.821832 - 8.74094e-5*(t - 52.14)
sos = signal.butter(6, [2000, 20000], "bandpass", fs=fs, output="sos")
def seg(n, t0, T, ch=0):
    s = t0 + (usbl_off(t0) if n == "USBL" else 0.0)
    i = int(round(s*fs)); x = R[n].data[i:i+int(T*fs), ch].astype(float)
    x = signal.sosfiltfilt(sos, x)
    # blank impulsive clicks: envelope > 4x running median
    e = np.abs(signal.hilbert(x)); med = np.median(e)
    bad = maximum_filter1d((e > 4*med).astype(np.uint8), 480) > 0   # +-2.5 ms around a click
    x[bad] = 0; return x, bad.mean()
def env(x, dec=48):   # 2 kHz envelope
    e = np.abs(signal.hilbert(x)); e = e[:len(e)//dec*dec].reshape(-1, dec).mean(1)
    return e - e.mean()
def xc(x, y, f_s, ml, phat=True, band=None):
    n = len(x); X = np.fft.rfft(x, 2*n); Y = np.fft.rfft(y, 2*n); f = np.fft.rfftfreq(2*n, 1/f_s)
    G = X*np.conj(Y)
    if band: G = np.where((f > band[0]) & (f < band[1]), G, 0)
    if phat: G = G/(np.abs(G)+1e-30)
    cc = np.fft.irfft(G); L = int(ml*f_s); cc = np.r_[cc[-L:], cc[:L+1]]
    return np.arange(-L, L+1)/f_s, cc
pairs = [("Rx902", "Rx01", 0.32), ("Rx902", "USBL", 0.36), ("Rx01", "USBL", 0.50)]
T = 4.0; ts = np.arange(780, 1016, 2.0)
fig, ax = plt.subplots(3, 2, figsize=(16, 12))
for pi, (a, b, ml) in enumerate(pairs):
    I1, I2 = [], []
    for t0 in ts:
        x, fa = seg(a, t0, T); y, fb = seg(b, t0, T)
        l, cc = xc(x, y, fs, ml); e = np.abs(signal.hilbert(cc)); I1.append(e/np.median(e))
        l2, c2 = xc(env(x), env(y), fs/48, ml, phat=False); I2.append(c2/np.std(c2))
    for j, (I, l_) in enumerate([(I1, l), (I2, l2)]):
        ax[pi, j].imshow(np.array(I).T, aspect="auto", origin="lower", extent=[ts[0], ts[-1]+2, l_[0]*1e3, l_[-1]*1e3],
                         vmin=(1 if j == 0 else 0), vmax=(10 if j == 0 else 5))
        ax[pi, j].set_title(f"{a}-{b} " + ("GCC-PHAT, clicks blanked" if j == 0 else "envelope xcorr"))
fig.tight_layout(); fig.savefig("/tmp/claude-1000/-home-bogas-workspace-boat-tracking/7b6002d4-07af-464f-a757-b9c5b5acefd6/scratchpad/cc5.png", dpi=70)
