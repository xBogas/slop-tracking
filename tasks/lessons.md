# Lessons

- **The experiment is fixed; only post-processing is ours.** I suggested
  changing the run plan (circles, standard zig-zag) and logging steering
  or onboard GPS. The user controls none of that and gets only pinger
  data. Rule: before suggesting a data-collection change, check that the
  user controls it; otherwise give a solution from the data available.

- **Derived results must follow the config.** I keyed the notebook's family-grid
  cache on ping indices only, and phase 2 scripts carried copied phase 1 numbers
  (positions, clock, c, 10 m depth) that went stale when `forfun.toml` changed.
  Rule: never paste a config value or an earlier result into code; read it from
  the config or the upstream output. Every cache gets a `stamp.key` of all its
  inputs, and every new step or notebook goes into `pipeline.stages()`.

- **Check what a "depth" is measured from.** The hydrophones were configured at "10 m depth"; the
  operator meant 10 m above the seabed. The bathymetry (13 m water) made "10 m deep, 3 m off the bottom"
  look odd and I noted it as a finding instead of asking. Rule: when a mooring depth sits oddly against
  the charted water depth, ask whether it is below the surface or above the seabed before building on it.
