# K: can the data check the hydrophone depth? (feasibility, 2026-10-22)

**Superseded:** the operator says the hydrophones are 10 m above the seabed (3.28 m / 2.65 m deep), not at
10 m depth. This note was written for the 10 m assumption; the refraction estimate below still holds.

Configured: 10 m on both buoys; charted water 13.3 m (Rx902) and 12.9 m (Rx01).

- **What the delays constrain.** In the image picture each multipath delay is a depth product over c R:
  S ∝ zs zr, B ∝ (D − zs)(D − zr), SB ∝ D (D + zs − zr), BS ∝ D (D + zr − zs). S and B are below 1/B here
  (note I). So the only usable pair, SB and BS, gives the effective water depth (from SB + BS) and
  **zr − zs** (from BS − SB), but not zr on its own. That is the depth/scale ambiguity of note D. The pinger
  depth zs is unknown too.
- **Data.** On 12 Rx902 pings with a unique SB and BS (pings 18–35, R ≈ 394 m), the flat-bottom algebra gives
  D_eff 12.4–13.2 m (chart: 9.3 m under the pinger, 11.8 m mean along the path) and zr − zs = 5.6–6.4 m. The same
  algebra applied to ray-traced delays over the sloping EMODnet bottom (true zr − zs = 7 m) returns 5.0 m. So
  the slope biases it by about −2 m, and only a ray-traced fit is meaningful. Rx01 has 2 such pings.
- **Refraction is not negligible.** First-order travel-time change along the straight image paths for a
  linear c(z), in µs relative to the direct path:

  | gradient | range | S | B | SB | BS |
  |---|---|---|---|---|---|
  | 0.05 m/s per m | 300–800 m | 15 to 40 | −6 to −15 | 10 to 25 | 5 to 13 |
  | 0.2 m/s per m | 300–800 m | 60 to 160 | −23 to −61 | 38 to 102 | 20 to 53 |

  That is as large as the 45 µs joint SB/BS misfit (note I) and the depth effects. With c/g = 7.6 km at
  0.2 m/s per m, rays sag about 2.6 m over 400 m in 13 m of water, so straight-ray (isovelocity) eigenrays
  are no longer exact. The profile is unknown.
- **Conclusion.** The data constrain zr − zs, the effective water depth and, with Rx01, zr902 − zr01. They do
  not constrain zr by itself. Doing even that properly needs a ray-traced fit with a sound-speed gradient as a
  free parameter (unknowns zs, zr, gradient, tide).
