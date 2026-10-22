# J: run times against the data

Source of `[[run]] start/end` in `forfun.toml`: `data/LocReceivers.xlsx`, ForFun row, run starts
15:59, 16:06 and 16:14. These are minute-resolution log times (0.665972 / 0.670833 / 0.676389 day), not fitted to
the data. The buoy files start at the first-run time by definition (sheet note).

## Checks (2026-10-22, phase 1 / phase 2 tracks)

- **The pinger never restarts.** Raw arrival intervals on every receiver are 1.001 s or exact multiples
  of it across gaps (Rx902 2.0018 and 3.0039 s at 411–419 s; Rx01 12.0193 s at 410–422 s). The phase 1 emission
  chain is continuous through 420 s: interval scatter ±0.1 ms, TOA residuals a few µs. Ping 360 (412.2 s),
  which no receiver heard, is a missed detection. The next ping falls exactly on the clock. Pings run
  continuously from 52.4 to 795.6 s.
- **Ping 360 is inside the turn between the runs** (387–444 s, course 175° → 45° → 133° → 22°). Around it only
  Rx902 or the USBL hears the pinger, and pings 353–359 are unstable: interval jumps of up to 0.76 ms,
  residuals of ±140 µs, and a 7.7 m track jump at 411 s. This is the hardest stretch of the run, not a boundary.
- **Data markers vs the log** (first ping at 52.4 s; the track passes each start fix at its closest point):

  | | log | data | marker |
  |---|---|---|---|
  | run 1 start | 0 s (15:59) | 102 s | passes the run 1 fix (14 m); straight leg from the 66–83 s turn |
  | run 2 start | 420 s (16:06) | 514 s | passes the run 2 fix (14 m); zig-zag turns start at 516 s and repeat every 40–60 s |
  | run 3 start | 900 s (16:14) | ~985 s (weak) | phase 2 track 29 m from the run 3 fix, inside the 926–984 s gap with no measurements |
  | pinger off | – | 795.6 s | last ping |

- **The differences agree with the log** to within its 1 min resolution: 412 s vs 420 s (run 1 → 2) and
  ~471 s vs 480 s (run 2 → 3). **The absolute times do not.** The boat reaches each start fix 85–100 s after the
  logged time. Either the log marks when a run was called, or its clock is ahead.

## Where the configured times matter

- `detect.py`: only the search span (min start to max end).
- `associate.py`: `run_ends` split the pinger-clock fits and the per-run USBL offset check at 420 s. The pinger is
  continuous, so the split adds freedom but is not physical (one Tp per run: 1.000389 / 0.999981 s).
- `run_phase1.py`: the `run` column, and the window each start fix searches for its closest epoch. That still
  finds 102 s and 514 s with the log windows.
- Pings 420–514 s (the transit after the turn) are labelled run 2.

## Decision (2026-10-22)

The config now trusts the data: run 1 starts at 102 s, run 2 at 514 s (it ends at 797 s, the pinger-off
time: last arrival 796.6 s plus the pulse), and run 3 at 985 s (weak). The log times are kept as comments.
`detect.py` and `chirp_timeline.ipynb` search from the file start, because the pinger is on before run 1.
`run_phase1.py --fixes` looks for each fix's closest epoch from 60 s before the run start (`FIX_MARGIN`),
because the start is itself the fix-passage estimate.
