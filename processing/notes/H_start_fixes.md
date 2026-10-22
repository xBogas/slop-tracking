# H: run start fixes (no time stamp) in the phase 1 graph

Input: `[[run]] lat/lon` in `forfun.toml` (boat positions near each run's start, no time
stamp; `config.Run.en`). Code: `fg_tracker.Config.pos_priors` (Gaussian prior on the
emitter at epoch k), `run_phase1.fix_priors` / `solve_with_fixes`; `run_phase1.py --fixes`
writes `out/phase1_fixes/` (`out/phase1` is untouched). `[priors] fix_sigma` (10 m) is a
placeholder.

Each fix is assigned to the epoch where the boat track passes closest to it inside that
run's window, moved to the pinger by the current tow offset, and the epochs are re-assigned
after each solve (they settle after one). Run 3's fix is outside the phase 1 window.

## Result (same config, sound speed prior 1529 +- 4)

| fix_sigma | Rx902 shift from GPS | Rx01 shift from GPS | fix miss run1 / run2 | cost |
|---|---|---|---|---|
| none | (+0.3, +0.2) | (-6.6, +26.0) | 8.7 / 12.0 m | 1091.2 |
| 10 m | (-1.1, +4.4) | (-9.4, +29.0) | 9.4 / 7.8 m | 1092.0 |
| 5 m | (-0.9, +7.8) | (-9.4, +32.1) | 9.2 / 5.3 m | 1094.0 |
| 3 m | (+0.1, +14.5) | (-7.4, +39.0) | 4.8 / 1.8 m | 1095.0 |

- Without them, the track passes 9-12 m from both fixes, about 1 sigma of the track.
- The fixes move the whole solution (buoys and track) north. The TOA cost barely changes
  (+0.8 at 10 m), because the common shift is a near-null direction of the data. The
  fixes are the only information on it, and they favour a 5-15 m shift north.
- Rx01 relative to Rx902 stays at about (-7, +25) m whatever fix_sigma is. The fixes do not
  settle the Rx01 question (position error vs timing delay); a common shift north
  increases Rx01's distance from its GPS position.
- Run 1 at its first ping (52 s) is 26 m from the fix. The boat drifts at ~0.3 m/s until
  ~100 s, and the closest pass is at ~102 s. Run 2's closest pass is at ~514 s, not the
  nominal 420 s.
