# Multipath: from nuisance to measurement

**Reading of the question.** "Multitracking" is taken to mean the extra arrivals and extra tracks that multipath creates: every hydrophone hears the direct path plus a surface reflection, a bottom reflection, and combinations of the two. In a TDOA correlogram each pair of paths draws its own track, so you see several tracks for one boat. The professor's suggestion is read as: each reflection is a copy of the first arrival shifted in time (in a spectrogram it is the same picture slid sideways), so shift the record against itself, line the copies up, and measure how far apart they are.

## What the shifting gives you

The delays between the copies depend only on the geometry under the receiver, not on when the signal was sent:

```
tau_p - tau_direct = ( sqrt(R^2 + (zr - z_p)^2) - sqrt(R^2 + (zr - zs)^2) ) / c
image depths z_p:  surface -zs,  bottom 2D - zs,  surface-bottom 2D + zs,  bottom-surface zs - 2D
far-field:         surface  ~ 2 zs zr / (c R)          bottom ~ 2 (D - zs)(D - zr) / (c R)
```

R is horizontal range to that receiver, zs the source depth, zr the hydrophone depth, D the water depth. So from **one hydrophone**, with no emission time, no clock sync and no other receiver, the multipath delays give:

1. **Horizontal range R** to that receiver. This is the big one. Delay goes as 1/R, so precision degrades with range (relative error equals relative delay error).
2. **Source depth zs.** The surface and bottom echoes scale differently with zs, so their ratio separates depth from range. In phase 1 this is the towed pinger's depth, which changes with boat speed. In phase 2 it shows the noise comes from near the surface.
3. **Water depth D** under the path (a constant over the run, so it can be estimated once).
4. **Range rate** from how the delays drift ping to ping (d tau / dt = -(tau / R) dR/dt), on top of Doppler.
5. **Which TDOA peak is the real one.** The autocorrelation at each receiver lists that receiver's own multipath lags, so the ghost tracks in the cross-receiver correlogram sit at predictable offsets from the direct-direct TDOA. That is the clean way to deal with the multiple tracks instead of only down-weighting outliers.

The spectrogram view and the correlation view are the same thing. For a chirp, an echo delayed by tau is the same sweep line shifted by tau in time (or by chirp-rate x tau in frequency). For broadband boat noise, two copies interfere and the spectrum gets a comb of fringes spaced 1/tau apart: the striations in the spectrogram. The Fourier transform of that spectrum (autocorrelation, or cepstrum if you take the log) has a peak at lag tau. As the boat moves, tau changes and the striations bend: that bending is range changing.

## Results on simulated data (`multipath_demo.py`)

Same truth track and buoy positions as `estimation/demo.py`. Flat bottom at 30 m, isovelocity water, hydrophones at 10 m, reflection coefficients -0.9 (surface, with random fading per ping) and 0.45 (bottom), five image paths.

**Phase 2, boat noise** (`noise_multipath.png`), one buoy, source at 1 m, 1 s windows, 0.5-15 kHz:

- Spectrogram shows clear striations; the whitened autocorrelation shows one ridge per pair of paths.
- Water depth estimated from the data: 30.00 m (true 30).
- Range from that single hydrophone: median error 2.4 m (0.4 %), 90th percentile 9 m, over 275-1240 m.
- If the water depth were assumed 1 m wrong, range is biased by about +7 %. Water depth must be estimated (or measured), not guessed.

**Phase 1, pinger** (`pinger_multipath.png`), 30-40 kHz LFM chirp, 5 ms, 1 ping/s, pinger at 5-10 m:

- Water depth estimated from the data: 30.0 m (true 30).
- Pinger depth, fitted jointly from both buoys: median error 0.12 m. It follows the towed body rising and sinking with boat speed (5.5 to 8.3 m here).
- Range from each single buoy: median error 0.6 to 1.0 m (0.1 %). The error distribution has two parts. Most pings are essentially exact. The rest land on a neighbouring ripple of the fit, 2 to 10 % off. A simple gate against the neighbouring pings kept 75 to 81 % of them. Even after the gate, the 90th percentile is 22 to 58 m, so the tracker's robust loss still has work to do.
- Position from the two buoy ranges alone, by intersecting the two circles: median error 1.2 m, 90th percentile 86 m. This fix uses no emission time, no clock sync and no shore array.

## Lessons from building it

- **Picking "the reflection" peak does not work.** There are more ridges than reflections (every pair of paths makes one), and cross-terms such as bottom-surface vs direct can be stronger than bottom vs direct. Fit the whole image-source model instead: for a candidate (R, zs, D), predict all pair lags (or the whole interference spectrum) and score it against the data. My first version picked peaks and was 15-20 % off.
- **For the pinger, the surface echo is within about 1/bandwidth of the direct path** (0.07-0.3 ms here vs 0.1 ms resolution for 10 kHz) and it is sign-flipped, so the two interfere and shift envelope peaks. An envelope/lag fit was off by hundreds of metres; a coherent fit of the received power spectrum |X(f)|^2 to the predicted comb x chirp spectrum fixes it. The power spectrum does not depend on the emission time either.
- **Pinger depth is the critical nuisance.** With the true depth, range from one buoy is exact to about 1 m. With the depth 0.5 m wrong, range is off by 5-10 % and sometimes much more. Fitting depth separately per buoy and per ping fails because depth and range trade off. It works because the depth is shared: both buoys hear the same ping, and the depth changes slowly, so their depth profiles are summed over +-3 pings. A cheap pressure logger on the pinger would remove this nuisance completely.
- **Phase 2 is easier than expected**, because the boat's noise source is so close to the surface that direct + surface act as one arrival, and the bottom echoes are well separated (0.6-5 ms).

## What this changes for the tracker

Each receiver gives a **range** (phase 1 and phase 2) that needs neither emission time nor clock sync. In the factor graph that is a new factor per receiver and epoch, `|p_k - r_i| = R_i,k`, with D (global) and zs (per epoch in phase 1, global in phase 2) as extra variables. Two such ranges from the buoys plus the shore bearing already fix the position. This matters most for phase 2, where TDOA alone left the track at about 22 m error because of geometry. I have not edited `estimation/`; that is the graph thread's code.

## Caveats before trusting it on real data

- Flat bottom and constant sound speed are assumed. A sloping bottom changes the bottom path per receiver (estimate D per receiver, or use bathymetry). A sound-speed profile bends the paths; with a strong thermocline the image model needs ray tracing.
- Hydrophone depth zr enters directly (range scales roughly with zr for the surface path and with D - zr for the bottom path). Log the buoy hydrophone depth (a pressure sensor is cheap) or estimate it.
- Wind-driven waves make the surface echo fade and jitter at 30-40 kHz. The bottom echoes carry most of the information in phase 1 here.
- The shore array sits in very shallow water (about 3 m in the demo geometry), where multipath is dense and the delays are tiny. It was not simulated; it is better used for bearing.
- The simulation uses the same reflection model for generating and fitting, apart from surface fading. Real reflection coefficients are unknown, so a real fitter should treat path amplitudes as nuisance parameters.

## Implications for the pinger design (goal 1)

- Bandwidth sets how well the surface echo separates from the direct path: use the full 10 kHz (LFM or a pseudo-random code), not a narrow tone.
- Keep the ping interval longer than the latest significant echo (tens of ms is plenty at these ranges), so echoes of one ping do not land on the next.
- A pinger depth that is not too shallow helps: the surface delay scales with zs.
- Put a depth (pressure) logger on the pinger if at all possible. It turns phase 1 multipath ranging from a joint fit into a direct readout.

Run: `python3 /mnt/project-files/multipath/multipath_demo.py` (numpy + matplotlib, about 12 minutes; most of it is the phase 1 grid search).
