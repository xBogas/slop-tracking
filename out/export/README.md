# Phase 1 measurements per receiver (out/export/)

Written by `processing/export.py` (re-run by `processing/pipeline.py`). Clock: buoy time, seconds since the buoy files' start (15:59:00 local, 2026-10-20). Phase 1 posterior: c = 1522.0 ± 4.0 m/s, USBL clock b_S = -11.84 ± 3.78 ms, drift -92.0 ± 10.6 ppm, array orientation -40.2 ± 1.0 deg (ENU).

`*_sigma` is the total sigma of a reading (full phase 1 posterior, with correlations); `*_sigma_random` is its ping-to-ping part (globals and buoy positions held fixed), which averages down. The rest is shared by every ping (clock offset, sound speed, buoy positions, orientation). Ranges and bearings are to the **pinger** (towed behind the boat), not the boat.

| file | rows |
|---|---|
| `first_arrival_Rx902.csv` | 732 |
| `first_arrival_Rx01.csv` | 691 |
| `first_arrival_USBL.csv` | 553 |
| `usbl_bearing_range.csv` | 553 |
| `multipath_Rx01.csv` | 808 |
| `multipath_Rx902.csv` | 1092 |

## first_arrival_<rx>.csv

| column | meaning |
|---|---|
| `ping` | ping index (emission number since the first ping) |
| `run` | run index (0 = run 1, 1 = run 2; pings before run 1 count as run 1) |
| `t_emit_buoy_s` | emission time of the chirp, buoy time [s] (phase 1 solution) |
| `t_emit_sigma_s` | its total sigma [s] (phase 1 marginal) |
| `t_emit_sigma_random_s` | its ping-to-ping sigma [s] (globals and buoy positions held fixed) |
| `toa_buoy_s` | first-arrival time (start of the chirp), synced to buoy time [s] |
| `toa_sigma_s` | total sigma of toa_buoy: toa_sigma_random (+) sync_sigma [s] |
| `toa_sigma_random_s` | receiver's empirical TOA noise (+) per-ping CRLB [s] |
| `toa_sigma_crlb_s` | per-ping CRLB alone [s] |
| `sync_offset_s` | USBL only: clock correction applied, toa_buoy = t_usbl_clock - sync_offset [s] |
| `sync_sigma_s` | USBL only: sigma of that correction, shared by all pings [s] (0 for the buoys: they define the clock) |
| `t_rx_clock_s` | arrival time on the receiver's own clock [s] (file time) |
| `snr_db` | matched-filter output SNR of the first path [dB] |
| `range_m` | one-way range to the pinger, c (toa_buoy - t_emit) [m] |
| `range_sigma_m` | total sigma of range [m]: full posterior of (t_emit, c, USBL clock) (+) toa noise |
| `range_sigma_random_m` | ping-to-ping sigma of range [m]: c (toa_sigma_random (+) t_emit_sigma_random) |
| `toa_resid_us` | toa minus the phase 1 prediction [us] |
| `outlier` | 1 if |toa_resid| > 5 toa_sigma_random |

## usbl_bearing_range.csv

| column | meaning |
|---|---|
| `ping` | ping index |
| `run` | run index |
| `t_emit_buoy_s` | emission time, buoy time [s] |
| `toa_buoy_s` | first arrival at the array centre, synced to buoy time [s] (blank if no TOA) |
| `range_m` | one-way range to the pinger [m] (as in first_arrival_USBL.csv) |
| `range_sigma_m` | total sigma [m] |
| `range_sigma_random_m` | ping-to-ping sigma [m] |
| `bearing_true_deg` | bearing from the array to the pinger, clockwise from true north [deg] |
| `az_enu_deg` | the same as an ENU azimuth: counter-clockwise from east [deg] (repo convention) |
| `bearing_sigma_deg` | total sigma: bearing_sigma_random (+) array orientation sigma (phase 1 alpha) [deg] |
| `bearing_sigma_random_deg` | ping-to-ping sigma (usbl.py per-ping sigma, floor 0.5 deg as in phase 1) [deg] |
| `bearing_resid_deg` | bearing minus the bearing to the phase 1 track [deg] |
| `bearing_used` | 1 if used by the phase 1 solve (usbl.py mask: good delays, no wall overlap) |
| `wall_overlap` | 1 if the wall echo overlaps the direct arrival |

## multipath_<buoy>.csv

| column | meaning |
|---|---|
| `ping` | ping index |
| `run` | run index |
| `t_emit_buoy_s` | emission time, buoy time [s] |
| `family` | path family (S surface, B bottom, SB, BS; 'S|B' = ambiguous: several windows hold it) |
| `t_direct_buoy_s` | arrival time of the direct path picked by multipath_paths.py, buoy time [s] |
| `t_path_buoy_s` | arrival time of this path, buoy time [s] |
| `dtau_us` | delay after the direct path [us] (clock-free measurement) |
| `sigma_us` | sigma of dtau [us]: CRLB (+) ping-to-ping scatter (see sigma_basis) |
| `sigma_crlb_us` | CRLB alone [us] |
| `sigma_basis` | scatter: this buoy and family; pooled: same family, other buoy; resolution: 1/(B sqrt 12) (no persistent detections of the family) |
| `level_db` | amplitude relative to the direct path [dB] |
| `snr_db` | SNR of the component [dB] |
| `pred_us` | predicted delay at the configured depths and tide [us] (first family if ambiguous) |
| `win_lo_us` | search window, low end [us] |
| `win_hi_us` | search window, high end [us] |
| `launch_deg` | predicted launch angle at the pinger [deg, + down] |
| `arrival_deg` | predicted arrival angle at the buoy [deg, + down] |
| `doppler_hz` | predicted Doppler minus that of the direct path [Hz] |
| `unique` | 1 if one family window holds it and no other component |
| `resolved` | 1 if >= 1/B after the direct path and >= 1/(2B) from other components |
| `persistent` | 1 if found on >= 2 of the +-2 neighbouring pings at the predicted delay change |
| `usable` | unique and resolved and persistent: the recommended measurements |
