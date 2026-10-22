# Turn bathymetry-predicted multipath into tracking information

With known bathymetry, multipath in a 30–40 kHz LFM link should be predicted, not just suppressed. A ray model or image model gives each surface and bottom path a delay, angle and Doppler. The receiver then uses those predictions three ways: to find the true direct path, to collect energy from the reliable paths, and as extra measurements in the tracking filter. The 10 kHz bandwidth gives about **100 µs (≈15 cm) delay resolution**. In 10–50 m water that is enough to separate first-order surface and bottom paths at a few hundred metres. They merge with the direct path at around 1–1.5 km, where only higher-order paths stay separate. The design has four parts. First, Doppler-compensate (time-scale) the matched filter, because LFM range–Doppler coupling biases the delay by several resolution cells at AUV speeds. Second, extract paths with a sparse or CLEAN-type estimator whose search windows come from the geometry. Third, use geometry only for delays and Doppler, and estimate amplitudes and phases from data, because 4 cm wavelengths make coherent multi-path replicas fragile. Fourth, feed every resolved path into a tracker as an image-source measurement with explicit data association. The literature shows this works with a single receiver. Single-receiver trackers using direct, bottom and bottom-surface delay differences have tracked boats to 3 km ([Watkins et al.](https://arxiv.org/html/2403.03384v2)), and an AUV has been localised at sea from one receiver plus bathymetry ([Dubrovinskaya, Casari, Diamant 2019](https://cris.haifa.ac.il/en/publications/bathymetry-aided-underwater-acoustic-localization-using-a-single-/)). Multipath makes depth observable without clock sync, and range can be recovered too once two path pairs are resolved. What is missing is a published accuracy figure for a Bellhop-in-the-loop tracker at 30–40 kHz. So treat the specific numbers below as engineering estimates to validate on your own data.

**How to read the flags.** "Sourced" means a finding from a cited paper. "Derived" means my own calculation from standard formulas: the image method, the Rayleigh reflection coefficient, LFM ambiguity, or Kirchhoff roughness, as in Jensen et al., *Computational Ocean Acoustics*, which was not re-fetched. A summary table of key numbers and their status comes before the conclusion.

## Geometry sets delays to 0.1 ms; the sea sets amplitudes

### BELLHOP via arlpy is the default predictor, with image sources as the fast path

At 35 kHz the wavelength is about 4.3 cm, so a water column tens of metres deep is thousands of wavelengths. That puts the problem in the ray and Gaussian-beam regime. BELLHOP (Porter, Acoustics Toolbox) computes the following ([BELLHOP manual](https://www.siplab.fct.ualg.pt/models/bellhop/manual/node2.html)):

- eigenrays and arrivals, with travel time, amplitude and angles;
- these over range-dependent bathymetry, sound-speed profiles (SSP) and surface altimetry.

arlpy's `uwapm` wrapper exposes three functions ([arlpy tutorial](https://arlpy.readthedocs.io/en/latest/_static/bellhop.html)):

- `create_env2d`, which takes the depth profile, SSP, bottom parameters and surface;
- `compute_arrivals`, which returns time of arrival, angle of arrival, and surface and bottom bounce counts;
- `arrivals_to_impulse_response`.

The image method is the closed-form special case for a flat (or constant-slope), isovelocity channel:

- 40 kHz-band channel modelling in about 15 m of water matched measured eigenray delays at 100 m range ([Chitre 2007](https://arl.nus.edu.sg/wp-content/publications/Chitre2007b.pdf)).
- RACE08 built its deterministic channel from image geometry alone, with no ray code ([Stojanovic, RACE08](https://www.mit.edu/~millitsa/resources/pdfs/nafplio.pdf)).

When bathymetry slopes strongly across the track, as at channel walls, harbour structures or a river mouth, bottom paths leave the vertical plane, and "the limits of Nx2D models have long been recognized". Bellhop3D addresses that case. Faceted 3D bathymetry still produces irregular fields ([Porter 2014](https://apps.dtic.mil/sti/pdfs/ADA616530.pdf)).

*Derived:* parabolic-equation models are a poor fit for arrival structure at this frequency. They need grids finer than 4 cm and handle steep multiples badly. Keep them as benchmarks.

### Image equations to implement (derived)

Consider a flat bottom at depth D, source depth z_s, receiver depth z_r and horizontal range R. The image depths of order m take four forms, which give the path length, delay, amplitude and grazing angle:

| Quantity | Expression |
|---|---|
| Image depths | 2mD + z_s − z_r, 2mD + z_s + z_r, 2(m+1)D − z_s − z_r, 2(m+1)D − z_s + z_r |
| Path length | L = √(R² + z_img²) |
| Delay | τ = L/c |
| Amplitude | ∝ (−1)^{n_s} R_b(θ)^{n_b} e^{−αL}/L |
| Grazing angle | θ = atan(z_img/R) |

In the small-angle limit, the extra delay relative to the direct path is:

- surface bounce: **Δτ_s ≈ 2z_s z_r/(cR)**
- bottom bounce: **Δτ_b ≈ 2(D−z_s)(D−z_r)/(cR)**

Sanity check against Chitre's geometry (z_s = 3 m, z_r = 2 m, R = 100 m): Δτ_s ≈ 80 µs, against the measured ≈ 0.09 ms and modelled ≈ 0.08 ms ([Chitre 2007](https://arl.nus.edu.sg/wp-content/publications/Chitre2007b.pdf)).

A sloping bottom with slope α rotates the ray by 2α at each bounce (derived). Upslope, multiples reach the critical angle sooner and die out. Downslope, more multiples survive. That fits the different spreads Silva et al. saw at upslope and downslope sites, although the sediment also differed between those sites ([Silva et al. 2010](https://www.siplab.fct.ualg.pt/pubs/asilva2.10.pdf)).

### Resolvability at B = 10 kHz collapses around 1 km

Derived rule: a path is resolvable from the direct path when **z_img² − (z_s − z_r)² > 2cR/B ≈ 0.3R** (m², with R in m).

- In 20 m water with both ends at 10 m depth, the surface and bottom paths are 0.67 ms behind the direct path at 200 m and 0.13 ms at 1 km. They merge with it beyond about 1.3 km.
- That symmetric geometry also makes the surface and bottom delays **exactly equal at every range**. Only arrival angle, or Doppler from vertical motion, can tell them apart.
- At km ranges, only paths with at least one extra full water-depth traversal (second-order and higher) stay resolvable.

Measured examples:

| Site | Observation |
|---|---|
| 100 m range ([Chitre](https://arl.nus.edu.sg/wp-content/publications/Chitre2007b.pdf)) | Later arrivals at 1.9–3.0 ms, 8–11 dB below the first |
| 1020 m range ([Chitre](https://arl.nus.edu.sg/wp-content/publications/Chitre2007b.pdf)) | Direct and surface paths merged into one fading first arrival |
| KAM11, 100 m water ([Badiey et al. 2014](https://data-ww3.ifremer.fr/BIB/Badiey_etal_JASA2014.pdf)) | Surface returns at 6–10 ms |
| Upslope sand site, ~25 kHz ([Silva et al.](https://www.siplab.fct.ualg.pt/pubs/asilva2.10.pdf)) | Delay spread up to ~70 ms |

### Some predictions hold to a cell; others do not

Delays are the reliable part of the prediction. Amplitudes are not.

**Sound speed (derived).** An error δc shifts the absolute delay by τ·δc/c, which is **0.44 ms per m/s at 1 km**, more than four resolution cells. Relative multipath delays have a much smaller base and barely move. Refraction can still decide which eigenrays exist at all, for example in a stratified river plume such as the Douro.

**Geometry and tide (derived).** These dominate the differential delays at short range:

- A 1 m tide error at D = 20 m and R = 200 m moves the bottom-path offset by about 0.13 ms.
- Each metre of receiver-depth error moves the surface-path offset by about 0.067 ms when z_s = 10 m.

**Bathymetry slope (derived).** A 1° slope error at 500 m moves a reflected ray by about 17 m, which is enough to create or delete an eigenray. The DEM should resolve the Fresnel scale, √(λL) ≈ 5 m at 500 m.

**Surface paths are the weak link.**

- *Derived:* with Kirchhoff coherent loss exp(−2k²σ²sin²θ), k ≈ 147 rad/m, a 10° grazing angle and rms height σ = 2 cm, the coherent part keeps 0.59 of its amplitude. At σ = 10 cm it keeps about 3×10⁻⁶.
- Above roughly sea state 1, surface arrivals should therefore be treated as fading taps with known mean delay but random amplitude and phase.
- *Sourced:* the field data agree. At 10–40 kHz in 30 m water, surface reflections "changed rapidly in arrival time, shape and amplitude" ([Nosal & Fedenczuk 2015](https://www.cetus.ucsd.edu/dclde/docs/pdfs/Tuesday/1-Nosal.pdf)).
- *Sourced:* in the 40 kHz band, Doppler spreads were 5–10 Hz and could not be predicted from the model ([Chitre 2007](https://arl.nus.edu.sg/wp-content/publications/Chitre2007b.pdf)).

**Bottom paths depend on sediment (derived).** The critical grazing angle is θ_c = arccos(c_w/c_b):

| Sediment | c_b (m/s) | θ_c |
|---|---|---|
| Sand | 1650–1750 | ≈ 25–31° |
| Silt or mud | 1520–1575 | ≈ 10–18° |

Low-order bottom bounces below θ_c are predictable in delay and moderately in amplitude. The ±3 dB figure is my judgement, not a measurement.

**Absorption.** Francois–Garrison is the standard formula ([Francois & Garrison 1982](https://data-ww3.ifremer.fr/BIB/Francois_Garrison_JASA1982.pdf)). The value of about 8–10 dB/km at 35 kHz is from recollection and should be computed, not trusted. Across km paths, absorption tilts the received spectrum by a few dB between 30 and 40 kHz.

## Build the receiver around Doppler first, geometry windows second

### Range–Doppler coupling is a bigger error than multipath at AUV speeds

An LFM's ambiguity function concentrates along the ridge f_d = kτ, with k = B/T. A matched filter evaluated only at zero Doppler therefore returns a delay offset proportional to Doppler, and a Doppler filter bank is needed to remove it ([Candan](https://users.metu.edu.tr/ccandan/pub_dir/LFM_receivers_Radar2008.pdf)).

Derived for this system, with T = 50 ms, B = 10 kHz and v = 2 m/s:

- Doppler at 35 kHz is about 47 Hz.
- The delay bias is f_d·T/B ≈ **0.23 ms (≈35 cm)**, two to three resolution cells.
- The bias differs per path, because each ray sees v·cosθ_p.

Both the time-compression product a·B·T ≈ 0.67 and the 29% fractional bandwidth point to a **wideband time-scale (resampling) Doppler bank**, not a frequency-shift bank. The practical fix is:

1. Resample the received signal by the common scale â from INS/DVL-free velocity, or from the peak of a coarse Doppler bank.
2. Subtract the residual per-path bias using the predicted ray angles.

This follows the wideband per-path-scale model of [Arunkumar & Murthy](https://ece.iisc.ac.in/~cmurthy/Papers/media/journal_arun_uwac.pdf).

An up/down-chirp pair cancels the bias, and HFM waveforms are Doppler-tolerant. Both are textbook results; the primary sources were not retrieved.

**Windowing.** Unweighted LFM sidelobes sit at about −13 dB, enough to hide a weak direct path. Hamming or Taylor weighting buys about −35 to −43 dB at a cost of about 1–1.5 dB SNR and a 1.3–1.5× wider mainlobe (standard figures). For path extraction, successive cancellation is better than heavy windowing, because windowing merges arrivals that are nearly coincident.

### Coherent within a path, incoherent across paths

**The channel-matched replica.** Replace the replica s(t) with ŝ(t) = s(t) ∗ ĥ_pred(t), where ĥ_pred is the predicted channel. This is passive time reversal with a model in place of a probe:

- The output is Q(t) ∗ x(t), where Q is the sum of channel autocorrelations.
- In a 6-element simulation, exploiting the multipath energy raised output SNR from 8.2 to 19.4 ([Jastrzębski 2007](https://acoustics.ippt.pan.pl/index.php/aa/article/download/1427/1243)).

At 35 kHz, though, coherent combination needs relative delay errors well below 1/f_c ≈ 29 µs, about 4 cm of path length. Waves, DEM error and SSP error rule that out in coastal water (derived).

**The recommended detector (derived design).** A **subspace / GLRT model-based matched filter**:

1. For each candidate position x, the geometry supplies the per-path delays τ_p(x) and Doppler scales a_p(x).
2. The statistic is ‖P_S r‖², the energy of r projected onto span{s((1+a_p)(t−τ_p))}. Path amplitudes and phases are free nuisance parameters.
3. Equivalently, sum per-path matched-filter envelopes inside windows of ±2–3/B around the predicted delays.

This needs delays accurate only to about 1/B ≈ 15 cm. That is realistic for the direct and first bottom paths, and marginal for the surface path above sea state 2. It mirrors the coherent-versus-incoherent trade-off of broadband matched-field processing.

The approach closest to it in the literature uses ray-modelled delay, amplitude, width and Doppler of the direct, surface and bottom pulses as features to pick out the direct arrival ([Li, Fu, Zou 2024](https://mdpi-res.com/d_attachment/jmse/jmse-12-00454/article_deploy/jmse-12-00454.pdf)):

- It identified the direct arrival with **98.45–100%** accuracy.
- It cut localisation RMSE from **4.87 m to 2.70 m**, using multiple sensors.
- Accuracy stayed at **98.5%** with 5 m/s sound-speed error and 4 m position error.
- Delay error was the most damaging, dropping accuracy to **95.3%** at 2 ms.

### Extract paths sparsely, with geometry as a prior

Underwater channels have "a few significant paths" in delay–Doppler. OMP followed by refinement, or sparse Bayesian learning (SBL), works on a dictionary of delayed and time-scaled replicas. On high-Doppler channels the Doppler-aware variants clearly win ([Arunkumar & Murthy](https://ece.iisc.ac.in/~cmurthy/Papers/media/journal_arun_uwac.pdf)). Gridded dictionaries leave an RMSE floor, so the final estimates need off-grid refinement, for example with SAGE ([Shutin & Fleury](https://vbn.aau.dk/files/61155739/Sparse_Variational_Bayesian_SAGE_Algorithm.pdf)).

Using bathymetry as a prior is my design proposal. No retrieved paper does it explicitly. Two versions:

- **Hard version: restrict the dictionary.** Place atoms only in windows τ̂_p ± 0.5–1 ms, b̂_p ± Δb around the predicted paths, at 25 µs spacing. Keep a coarse grid elsewhere to catch unmodelled scatterers, such as harbour walls and hulls.
- **Soft version: set SBL priors from geometry.** Start the hyperparameters γ_i large near predicted paths and small elsewhere. This degrades gracefully when the prediction is wrong.

### Do not trust the biggest peak

In shallow water, "simple matched filtering was unreliable because multipath arrivals often were stronger than the direct path". Bounding the search window improved direct-path estimates ([Zoksimovski & de Moustier 2004](https://scholars.unh.edu/ccom/296)).

A practical direct-path picker (derived) runs CLEAN on the matched-filter output:

1. Find the peak, interpolate it to sub-sample delay, subtract a scaled replica autocorrelation, and repeat until the peak falls below a CFAR threshold.
2. Take the direct path as the **earliest** component above a low threshold (for example −15 dB relative to the strongest) inside the geometry-predicted direct-path window.
3. If nothing qualifies, flag "direct path missing" and fall back to the all-paths model-based range.

When arrivals overlap within 1/B, which is common near boundaries, there are three super-resolution options, all of which need Doppler compensation first:

- SAGE initialised from the predicted delays ([Fleury et al. 1999, via Shutin & Fleury](https://vbn.aau.dk/files/61155739/Sparse_Variational_Bayesian_SAGE_Algorithm.pdf));
- frequency-domain ESPRIT on H(f) = R(f)/S(f) across 30–40 kHz;
- off-grid SBL.

### A USBL head separates paths by angle, but only at steep angles

SAGE natively estimates delay and angle jointly per path ([Shutin & Fleury](https://vbn.aau.dk/files/61155739/Sparse_Variational_Bayesian_SAGE_Algorithm.pdf)). With predicted elevations θ_D, θ_S and θ_B, a wideband LCMV beamformer can be built (derived): constraints C = [v(θ_D), v(θ_S), v(θ_B)] and gains g = [1, 0, 0], so unit gain on the direct path and nulls on the reflections, with diagonal loading for robustness.

A few-centimetre USBL aperture gives beams tens of degrees wide. Nulling therefore works only where the reflected angles are well separated from the direct one, meaning short range or a large depth difference. At km ranges in shallow water every ray is within a few degrees of horizontal, so delay-domain methods have to do the work. Steer per frequency bin, because 29% fractional bandwidth is too wide for narrowband phase steering. Even when angle cannot null a path, it remains a useful gate for association.

## Every resolved reflection is a virtual anchor with known position

### Image sources make depth, and sometimes bearing, observable from one hydrophone

Each specular reflection behaves like a direct path from a mirrored source or receiver. Delay *differences* cancel the emission time, so no synchronized clock is needed. The working tracker of [Watkins et al.](https://arxiv.org/html/2403.03384v2) uses:

- path lengths q_k(p) = √((d − i_k(h))² + r²);
- imaged receiver depths i_DP = h, i_BB = 2z − h and i_BSB = 2z + h;
- TDOA measurements g_l(p) = |q_{l1} − q_{l2}|/c with σ = 0.5 ms.

From the derived small-angle form, Δτ_s ≈ 2dh/(cR) constrains only the product dh/R. A second pair, such as bottom versus direct, is needed to separate range from depth. When bathymetry varies with azimuth, bearing becomes observable too:

- a sperm whale was tracked in 3D from one bottom hydrophone on a slope at AUTEC ([Tiemann 2008](https://jcaa.caa-aca.ca/index.php/jcaa/article/view/1993));
- Nosal frames the same idea as a "virtual receiver" ([Nosal & Fedenczuk](https://www.cetus.ucsd.edu/dclde/docs/pdfs/Tuesday/1-Nosal.pdf)).

The closed-form passive multipath ranging error budget of [Dubrovinskaya, Nissen, Casari 2016](https://dspace.networks.imdea.org/bitstream/20.500.12761/257/1/accuracy-passive-multipath-6.pdf):

| Error source | Range error |
|---|---|
| Fixed c = 1500 m/s, 200–1000 m (Bellhop simulation) | 2–30 m |
| Effective sound speed c_eff from path pairs | < 1% |
| 1% water-depth error | 2–3% |
| ±5% TDoA error | ~5% |
| Receiver depth estimated jointly | ~3% typical, up to 15% for downward refraction |

Errors grow sharply near the surface or bottom, where arrivals overlap.

**Prior caution: depth is the weak coordinate.** The low-frequency model-based single-hydrophone work of Jesus, Porter et al. (300–800 Hz, 5.5 km) gives these results ([Jesus et al.](https://www.siplab.fct.ualg.pt/pubs/sjesus2.98.pdf)):

- Range: 5.2 ± 0.55 km.
- Depth: only **81 ± 35 m** in a 135 m guide.
- AIC found **4–5 resolvable paths against 48 predicted**.

Expect the same gap between modelled and usable paths at 35 kHz.

### The tracker needs explicit multipath data association

The standard formulation:

- one measurement function per propagation mode, h_k(x);
- a latent association variable mapping detected peaks to modes, misses or clutter.

Watkins et al. ([arXiv 2403.03384](https://arxiv.org/html/2403.03384v2)):

- State: [range, depth, range-rate].
- Modes: three path pairs, with an exclusivity constraint ψ(a) so no two modes claim the same peak.
- Clutter: Poisson false alarms.
- Inference: particle sum-product belief propagation with 10⁴ particles.
- Field result, Cinque Terre in 65–73 m water: low RMSE for a workboat within 500 m. A cargo ship at up to 3 km degraded after about 250 s, when misses and false alarms rose.

Meyer & Gemba ([arXiv 2109.06818](https://ar5iv.arxiv.org/html/2109.06818)):

- Associate vertical-array DOAs with BELLHOP-predicted eigenray DOAs for four modes.
- Set detection probability d_k = 0 for geometrically impossible paths.
- Their probabilistic focalization outperformed matched-field processing on SWellEx-96 data.

The over-the-horizon-radar lineage supplies ready templates for the association step ([Huang, Shi, Song 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6471521)):

- MPDA (Pulford & Evans);
- multipath IPDA, with L!/(L−φ)! assignment hypotheses per cell;
- MD-JPDA, MD-MHT and multipath Bernoulli or PHD filters.

Labeled-RFS filters go further and estimate the reflectors jointly with the targets ([Ding et al.](https://arxiv.org/html/2602.03464v2)).

### Information gain scales with each path's SINR and delay sensitivity

UWB radio provides the clean theory ([Leitinger et al.](https://ar5iv.labs.arxiv.org/html/1409.1467)):

- Model: τ_k = ‖p − p_k‖/c + ε for virtual anchors p_k.
- Diffuse multipath is treated as coloured Gaussian interference.
- The position Fisher information is I_θ = Jᵀ I_ψ J, with J = ∂ψ/∂θ.
- Each specular component adds information in proportion to its SINR.

*Derived by analogy, not shown underwater:* the acoustic gain from path k scales roughly with SINR_k · (B_rms · ∂τ_k/∂p)². Stable bottom bounces off smooth sediment are worth the most. Rough-sea surface bounces are worth little and should carry inflated variance and a lower detection probability.

### Precompute arrival tables; run the image method online

For embedded use, the per-query cost matters:

- **Image method:** about a microsecond per image (derived). Practical inside a DUNE task for well-mixed, gently sloping sites.
- **BELLHOP tables (derived design):** for stratified or strongly range-dependent sites, run BELLHOP offline per bearing sector (every 2–5°) on a grid of about 5–10 m in range by 0.5 m in depth. Store (τ, |A|, angles, n_s, n_b) for each arrival.
- **Interpolate by path family, not arrival index.** Families are identified by bounce counts, and arrival ordering changes across the grid.
- **Doppler:** compute online as f₀ v·û_i/c from the tabulated angles and the navigation velocity.

Two sourced results support the approach:

- The channel needs far fewer snapshots than the signal rate: 501 were reduced to 13 ([Silva et al. 2010](https://www.siplab.fct.ualg.pt/pubs/asilva2.10.pdf)).
- Post-processing standard ray output for steady motion is "much more efficient" than full VirTEX simulation ([Peterson & Porter 2013](https://manuscript.isc.ac/Inventory/49/7827.htm)).

A single-receiver AUV localiser that matches measured channel responses to a precomputed Bellhop database, followed by a Viterbi-like tracker, has been validated at sea. The abstract reports no error figure ([Dubrovinskaya, Casari, Diamant 2019](https://cris.haifa.ac.il/en/publications/bathymetry-aided-underwater-acoustic-localization-using-a-single-/)).

## Recommended end-to-end pipeline

The stages below combine the sourced building blocks above. The pipeline as a whole is my synthesis and has not been tested as a system.

| # | Stage | What to do | Basis |
|---|---|---|---|
| 0 | Environment prior | Gather a multibeam or EMODnet DEM at about 5 m resolution, a tide gauge or model, a CTD-cast SSP and the sediment class. Draw an ensemble over tide offset, SSP EOFs, sediment class and sea surface. | Ensemble/focalization practice ([Hodgkiss 2012](https://apps.dtic.mil/sti/pdfs/ADA574869.pdf)); member sizes are undetermined |
| 1 | Arrival prediction | Use the image method online for flat, mixed sites. Otherwise use BELLHOP (arlpy) tables per bearing, clustered by (n_s, n_b) family into mean delay, spread and existence probability. Add Bellhop3D where slopes cross the track. | Sourced tools, derived table design |
| 2 | Waveform and front end | Taper the chirp edges with a Tukey window. Consider an up/down LFM pair or HFM. Resample by â from navigation velocity or a coarse time-scale bank. | Derived/textbook |
| 3 | Pulse compression | Matched filter against the time-scaled replica, with optional mild receive weighting. Correct the residual per-path coupling bias using predicted angles. | [Candan](https://users.metu.edu.tr/ccandan/pub_dir/LFM_receivers_Radar2008.pdf) + derived |
| 4 | Path extraction | Run CLEAN or OMP restricted to the geometry windows plus a coarse catch-all grid, then refine with SAGE, ESPRIT or off-grid SBL where paths overlap within 1/B. Add angle from the USBL where available. | [Arunkumar & Murthy](https://ece.iisc.ac.in/~cmurthy/Papers/media/journal_arun_uwac.pdf), [Shutin & Fleury](https://vbn.aau.dk/files/61155739/Sparse_Variational_Bayesian_SAGE_Algorithm.pdf) |
| 5 | Direct-path decision | Take the earliest qualifying component in the predicted window. If there is none, use the GLRT all-paths range and flag the direct path missing. | [Zoksimovski & de Moustier](https://scholars.unh.edu/ccom/296), derived GLRT |
| 6 | Measurement model | Per mode, τ_k = ‖p_k^img(p) − p_rx‖/c_eff + ε, or a table lookup T_k(p) + ε. Use differences to drop ε without clock sync; keep absolute τ_D with a synchronized one-way-travel-time clock. Set d_k from predicted loss and grazing angle, d_k = 0 for impossible modes, and inflate σ for the surface path. | [Watkins](https://arxiv.org/html/2403.03384v2), [Meyer & Gemba](https://ar5iv.arxiv.org/html/2109.06818) |
| 7 | Tracker | Particle filter with belief-propagation or JPDA association over modes plus clutter and miss hypotheses. State: position, velocity, tide/depth offset, c-bias, receiver depth bias (sequential focalization). Hand over to a UKF once gates are tight, and fall back to direct-path-only updates when no reflection passes the gate. | Sourced templates, derived state augmentation |
| 8 | Diagnostics | Log association probabilities per mode. A persistent mismatch indicates environmental error, such as a wrong tide or DEM bias. | Derived |

### Key numbers and their status

| Quantity | Value | Status |
|---|---|---|
| Delay resolution 1/B | 100 µs ≈ 15 cm | Derived |
| LFM coupling bias (2 m/s, T = 50 ms) | ≈ 0.23 ms ≈ 35 cm | Derived |
| Surface/bottom merge with direct (20 m water, mid-depth) | ≈ 1.3 km | Derived |
| Delay shift per 1 m/s sound-speed error at 1 km | 0.44 ms | Derived |
| Coherent surface factor at σ = 10 cm, 10° grazing | ~3×10⁻⁶ | Derived |
| Absorption at 35 kHz | ~8–10 dB/km | Recollection, compute it |
| Surface Doppler spread, 40 kHz band | 5–10 Hz | Sourced ([Chitre](https://arl.nus.edu.sg/wp-content/publications/Chitre2007b.pdf)) |
| Range error per 1% water-depth error | 2–3% | Sourced ([Dubrovinskaya 2016](https://dspace.networks.imdea.org/bitstream/20.500.12761/257/1/accuracy-passive-multipath-6.pdf)) |
| Direct-path identification / RMSE gain | 98.45–100%, 4.87 → 2.70 m | Sourced ([Li et al.](https://mdpi-res.com/d_attachment/jmse/jmse-12-00454/article_deploy/jmse-12-00454.pdf)) |
| Bottom-path amplitude accuracy | ±3 dB | Judgement |

### Gaps to verify before relying on this

The evidence is weakest at the operating point you care about:

- **No accuracy figure at 30–40 kHz.** No retrieved paper reports accuracy for a Bellhop-in-the-loop or image-source multipath tracker at 30–40 kHz. Every model-based field result is at ≤1 kHz, at abstract level only, or for boats tracked at 1–3 km in deeper water.
- **No underwater CRB.** No underwater Cramér–Rao bound for single-hydrophone multipath ranging was found. The SINR-weighted information argument is borrowed from radio.
- **Untested design proposals.** No source uses bathymetry-predicted delays as SBL priors or as a restricted dictionary. Several building blocks also rest on textbook formulas whose primary sources were not fetched:
  - HFM bias formulas;
  - window figures;
  - the BELLHOP arrivals-file format and `.brc`/`.trc` reflection options, to be checked in Porter's user guide on oalib.
- **Unquantified effects.** No measurements were found for:
  - how often cross-slope bathymetry creates resolvable out-of-plane arrivals;
  - ensemble size for arrival prediction;
  - BELLHOP runtime at these frequencies;
  - a wind-speed versus surface-coherence curve at 30–40 kHz.
- **Unsearched groups.** LSTS/FEUP, Chitre/UnetStack and SiPLAB work on multipath-exploiting navigation was not searched properly and deserves a dedicated look. Soares & Jesus on broadband matched-field processing is one known lead in that literature.

## Conclusion

The question is not whether to remove multipath but which paths to trust for which job. Geometry predicts delays to within a resolution cell for the direct and first bottom paths at hundreds of metres, so those paths should drive both the receiver windows and the tracker's measurement model. Surface paths are the opposite case at 35 kHz. Their mean delay is known, but their amplitude and phase are effectively random. Use them as low-confidence, incoherent measurements and do not build replicas from them. The bandwidth sets a hard ceiling. Beyond about a kilometre in coastal depths, first-order reflections fold into the direct arrival. From there the extra information comes only from higher-order paths, from angle when the geometry is steep, or from a longer baseline that changes the image geometry over time. The tracker's state augmentation helps there because it turns tide, DEM and sound-speed errors into estimated states.

For a DVL-free AUV the second-order consequence matters most. A tracker that associates multipath modes is also an environmental sensor. Persistent association bias on bottom paths measures water-depth or tide error. The consistency between surface and bottom paths constrains depth-sensor bias and effective sound speed. These are exactly the slowly varying nuisance terms that degrade a learned velocity and current estimator. The highest-value next step is to replay logged 30–40 kHz matched-filter outputs against image-method predictions on known LSTS tracks. One experiment would then answer three things: path resolvability, surface-path stability, and the real information gain over direct-path-only ranging.