# Boat motion models (planar, kinematic)

Files: `boat_model.py` (models), `demo.py` (scenario + plots),
`demo_trajectory.png`, `demo_truth.csv` (synthetic truth, 20 Hz).
Run with `uv run python motion_model/demo.py`.

## Conventions
- World: x east, y north. Heading ψ from +x, counter-clockwise positive.
- Turn rate r / ω CCW positive (to port).

## Synthetic truth (`demo.py`)

A kinematic track, not a dynamic model: we have no throttle, rudder or hull
data to fit one. Speed through water and path curvature are scripted per
segment (straight run, port turn, fast straight, zig-zag, slow-down, tight
starboard circle, wander) with first-order lags (2.5 s on speed, 1 s on turn
rate), integrated with `ct_track` (the CT model below), plus a constant
0.2/−0.1 m/s current. Heading is the course through the water, so there is
no sideslip. `demo_truth.csv` columns: `t, x, y, psi, u, r, vx_ground,
vy_ground, pinger_x, pinger_y`.

## Towed pinger (`towed_body_track`)

The pinger buoy is modelled as a point pulled on a taut 5 m horizontal line
from the stern (tractrix / pure pursuit); it stops when the line is slack.
In the demo the pinger sits 7.5–8.0 m behind the boat's centre
(3 m to the stern + 5 m line) and cuts corners in turns. So the phase 1
tracker estimates the **pinger** track; turning it into a boat track needs
this offset model (or just an along-course offset of ~8 m when going straight).
Line catenary, buoy drag and the 10 m depth are not modelled; only the
horizontal geometry is.

### Pinger track -> boat track (`tow_point_from_body`, `boat_centre_from_tow`)

In the pure-pursuit model a taut line always lies along the towed body's
velocity (the body moves straight at the tow point), so the inverse is

```
tow point  Q = P + L V/|V|                (P, V: pinger position, ground velocity)
boat       C = Q + stern_offset VQ/|VQ|   (VQ: tow point ground velocity)
```

`boat_from_towed_body(P, V, t, L=5, stern_offset=3)` does both and returns
`(Q, C)`. Where |V| < `v_min` (stopped, slack line) the last direction is
held; the tow point is then only known to within L. The second step uses
course over ground as the heading, so crab angle (current, sideslip in
turns) shows up as about `stern_offset x crab angle`. On the demo truth
(`tests/test_tow.py`): tow point < 1 mm and boat centre < 0.1 m on straight
legs (also at the 1 Hz ping rate); in turns the tow point is within 0.08 m
and the boat centre within 0.8 m (95 %: 0.78 m), from the stern swinging out.
The demo has no sideslip, so real turns will be somewhat worse.

## Kinematic model for the tracker (`ct_*`)

The tracker never sees throttle or rudder, so it uses a **coordinated-turn
(CT)** model on what the acoustics actually observe:

```
z = [x, y, χ, U, ω]      χ course over ground, U speed over ground, ω turn rate
x+ = x + U/ω (sin(χ+ωΔt) − sinχ)
y+ = y + U/ω (cosχ − cos(χ+ωΔt))
χ+ = χ + ωΔt,  U+ = U,  ω+ = ω       (+ random accel σa and turn accel σω)
```

- Course over ground, not heading, so sideslip and current are absorbed.
- `ct_predict` is the exact discrete step, written with sinc terms so it is
  smooth through ω = 0 (no branch, so finite-difference derivatives w.r.t. ω
  stay correct at ω ≈ 0); it and `ct_process_noise` accept `(..., 5)` arrays,
  `ct_jacobian` is for an EKF, `ct_process_noise` gives Q, `ct_track`
  integrates given speed and turn-rate histories.
- On the demo pinger track, CT predicts 1 s ahead with median error 0.04 m
  and 95th percentile 0.43 m; the errors are at turn-rate changes, as expected.
- For phase 2 the same CT model applies to the boat's noise centre
  (propeller/engine, near the stern), with no tow offset.
