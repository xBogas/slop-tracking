"""
Batch factor-graph smoother for the acoustic boat tracker (post-processed).

Solved as one sparse nonlinear least-squares problem (Levenberg-Marquardt
with a sparse direct solve, Huber loss by IRLS on the measurement and
motion factors; priors stay Gaussian). Every factor returns a
whitened residual (error / sigma), so the cost is the sum of squared
factors, exactly the MAP estimate of the factor graph.

Variables
  Z[k]   = [x, y, chi, U, omega]  emitter CT state at epoch k
           (pinger in phase 1, noise centre in phase 2), from motion_model/
  T[k]   = ping emission time [s]                (phase 1 only)
  B[k,j] = horizontal position of buoy j at epoch k (drifts, random walk)
  g      = globals [b_S, alpha, s, c, cx, cy, d_S]   (index = GLOBALS order)
           b_S   shore (USBL) clock offset w.r.t. the synchronized buoys [s]
           alpha shore array orientation error (bearing bias) [rad]
           s     pinger clock skew (fractional)             (phase 1 only)
           c     sound speed [m/s]
           cx,cy water current [m/s]            (motion="water" only)
           d_S   shore clock drift [s/s]; the shore offset at epoch k is
                 b_S + d_S t_k with t_k = k Tp (= T[k] - T[0] up to s d_S)

Receivers: index 0..nb-1 are the buoys, index nb is the shore array.

Factors
  motion      Z[k+1] ~ ct_predict(Z[k], dt), cov ct_process_noise(...)
              motion="ground": Z is course/speed over ground (current absorbed)
              motion="water":  Z is course/speed through water and the
                               position also advects with (cx, cy) * dt
  ping chain  T[k+1] - T[k] = Tp (1 + s)                   (phase 1)
  TOA         a_ik = T[k] + |p_k - r_ik| / c + b_i(k)       (phase 1)
  TDOA        d_ij = (|p - r_i| - |p - r_j|)/c + b_i(k) - b_j(k)  (phase 2)
  bearing     theta_k = atan2(p - shore) - alpha
  mp range    R_ik = |p_k - r_ik| horizontal   (single-receiver multipath range)
  range rate  rr_ik = (p_k - r_ik) . v_k / |p_k - r_ik|   (3-D slant geometry,
              horizontal emitter velocity over ground, receivers static)
  buoy drift  B[k+1] - B[k] ~ N(0, q_b dt),  B[0] ~ nominal (mooring) prior
  start       Z[0, :2] ~ N(start, sigma)   (optional, cfg.start_prior)
  positions   Z[k, :2] ~ N(p_k, sigma_k)   (optional, cfg.pos_priors)
  priors      on the globals (sync signal, sound-speed profile, ...)
  with b_i(k) = 0 for buoys, b_S + d_S t_k for the shore.

Measurements (meas dict; all (K, ...) arrays are per epoch, masks are bool)
  phase 1   toa (K, nr), toa_mask (K, nr), toa_sig (K, nr) optional [s]
  phase 2   pairs [(i, j), ...], tdoa (K, npairs), tdoa_mask (K, npairs),
            tdoa_sig (K, npairs) optional [s]
  optional  bearing (K,), bearing_mask (K,)                     [rad]
            mp_range, mp_range_mask, mp_range_sig (K, nr)      [m]
            rr, rr_mask, rr_sig (K, nr)  range rate d|p - r|/dt [m/s]
  A missing *_sig falls back to the scalar Config sigma.

Missing detections are simply absent factors (mask), so the graph handles
dropouts (even epochs with no receiver at all) natively; the motion and
ping-chain factors carry the state through.
"""
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.optimize._numdiff import group_columns
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "motion_model"))
from boat_model import ct_predict, ct_process_noise  # noqa: E402

NX = 5  # CT state size
GLOBALS = ("b_S", "alpha", "skew", "c", "cx", "cy", "d_S")
NG = len(GLOBALS)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def spd_lu(A):
    """Sparse LU of a symmetric positive definite matrix: symmetric mode
    (no pivoting) with minimum-degree ordering on A + A^T. ~7x faster than
    the default COLAMD here (fill stays near the time band + dense globals)."""
    return splu(csc_matrix(A), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                options=dict(SymmetricMode=True))


@dataclass
class Geometry:
    shore: np.ndarray                 # (3,) shore array x, y, z (z negative = depth)
    buoys_nominal: np.ndarray         # (nb, 2) nominal buoy x, y
    buoy_depth: float = -10.0          # z of the buoy hydrophones: one value, or one per buoy (nb,)
    src_depth: float = -10.0          # pinger / propeller depth (noise ~ -1 m)


@dataclass
class Config:
    mode: str = "pinger"              # "pinger" (phase 1) or "noise" (phase 2)
    motion: str = "ground"            # "ground" or "water" (estimates current)
    Tp: float = 1.0                   # ping interval [s] (phase 1) / epoch step (phase 2)
    # motion factor (CT model of motion_model/)
    sigma_a: float = 0.5              # along-track accel [m/s^2]
    sigma_w: float = np.deg2rad(3.0)  # turn accel [rad/s^2]
    q_reg: tuple = (0.3, 0.3, np.deg2rad(1.0), 0.05, np.deg2rad(0.2))  # extra Q (model error)
    # measurements (defaults when meas has no per-entry sigma)
    sig_toa: float = 50e-6            # [s]
    sig_tdoa: float = 0.2e-3          # [s]
    sig_bearing: float = np.deg2rad(2.0)
    sig_chain: float = 20e-6          # emission jitter [s]
    sig_mp: float = 5.0               # multipath horizontal range [m]
    sig_rr: float = 0.1               # Doppler range rate [m/s]
    # buoys
    sig_buoy0: float = 15.0           # mooring prior [m]
    q_buoy: float = 0.05              # drift random walk [m/sqrt(s)]
    # globals: prior means and sigmas  [b_S, alpha, s, c, cx, cy, d_S]
    g_mean: tuple = (0.0, 0.0, 0.0, 1500.0, 0.0, 0.0, 0.0)
    g_sig: tuple = (0.2e-3, np.deg2rad(10.0), 20e-6, 3.0, 1.0, 1.0, 50e-6)
    start_prior: tuple = None         # (x, y, sigma) on the first epoch, or None
    pos_priors: tuple = ()            # ((k, x, y, sigma), ...) on the emitter at epoch k
    huber: float = 3.0                # robust loss scale, in sigmas
    loss: str = "cauchy"              # measurement rows: "huber" or "cauchy" (see solve)
    nb: int = 2                       # number of buoys
    # initialisation (initial_guess)
    region: tuple = None              # (xmin, xmax, ymin, ymax) grid-search box;
                                      # None: receivers' bounding box +- 1.5 km
    grid_step: float = 25.0           # [m]
    reseed_gap: int = 10              # re-run the grid search after this many epochs without a fix
    u_max: float = 15.0               # plausible max speed [m/s] (init gating)


class FactorGraph:
    """Builds the variable layout, residual function and Jacobian sparsity."""

    def __init__(self, cfg: Config, geo: Geometry, K: int, meas: dict):
        self.cfg, self.geo, self.K = cfg, geo, K
        self.pinger = cfg.mode == "pinger"
        nb = cfg.nb
        self.nr = nr = nb + 1
        if len(cfg.g_mean) != NG or len(cfg.g_sig) != NG:
            raise ValueError(f"g_mean/g_sig need {NG} entries {GLOBALS}")
        self.tk = np.arange(K) * cfg.Tp          # epoch time for the shore drift
        self.m = m = dict(meas)
        # optional factors and per-entry sigmas, normalised to full arrays
        m.setdefault("bearing", np.zeros(K))
        m.setdefault("bearing_mask", np.zeros(K, bool))
        if self.pinger:
            m["toa_sig"] = np.broadcast_to(m.get("toa_sig", cfg.sig_toa), (K, nr)).astype(float)
            m["toa"] = np.where(m["toa_mask"], m["toa"], 0.0)
        else:
            npq = len(m["pairs"])
            m["tdoa_sig"] = np.broadcast_to(m.get("tdoa_sig", cfg.sig_tdoa), (K, npq)).astype(float)
            m["tdoa"] = np.where(m["tdoa_mask"], m["tdoa"], 0.0)
        self.has_mp = "mp_range" in m
        self.has_rr = "rr" in m
        for key, sig in (("mp_range", cfg.sig_mp), ("rr", cfg.sig_rr)):
            if key in m:
                m[key + "_mask"] = np.asarray(m.get(key + "_mask", np.isfinite(m[key])), bool)
                m[key] = np.where(m[key + "_mask"], m[key], 0.0)
                m[key + "_sig"] = np.broadcast_to(m.get(key + "_sig", sig), (K, nr)).astype(float)
        # variable layout
        o = 0
        self.iZ = o + np.arange(K * NX).reshape(K, NX); o += K * NX
        if self.pinger:
            self.iT = o + np.arange(K); o += K
        self.iB = o + np.arange(K * nb * 2).reshape(K, nb, 2); o += K * nb * 2
        self.iG = o + np.arange(NG); o += NG
        self.n = o
        self._build_sparsity()

    # ------------------------------------------------------------------ utils
    def unpack(self, v):
        Z = v[self.iZ]
        T = v[self.iT] if self.pinger else None
        B = v[self.iB]
        g = v[self.iG]
        return Z, T, B, g

    def receivers(self, B):
        """(K, nr, 3) receiver positions per epoch: buoys then shore."""
        K, nb = self.K, self.cfg.nb
        R = np.empty((K, nb + 1, 3))
        R[:, :nb, :2] = B
        R[:, :nb, 2] = self.geo.buoy_depth
        R[:, nb] = self.geo.shore
        return R

    def emitter(self, Z):
        return np.column_stack([Z[:, :2], np.full(self.K, self.geo.src_depth)])

    def ranges(self, Z, B):
        return np.linalg.norm(self.emitter(Z)[:, None, :] - self.receivers(B), axis=2)   # (K, nr)

    def clock(self, g):
        """(K, nr) receiver clock offsets: 0 for buoys, b_S + d_S t_k for shore."""
        b = np.zeros((self.K, self.nr))
        b[:, -1] = g[0] + g[6] * self.tk
        return b

    # -------------------------------------------------------------- residuals
    def residuals(self, v):
        c, m = self.cfg, self.m
        Z, T, B, g = self.unpack(v)
        alpha, s, cs, cx, cy = g[1:6]
        dt = c.Tp
        out = []

        # motion factors, whitened by chol of Q at the current linearization
        pred = ct_predict(Z[:-1], dt)
        if c.motion == "water":
            pred[:, :2] += np.array([cx, cy]) * dt
        e = Z[1:] - pred
        e[:, 2] = wrap(e[:, 2])
        Q = ct_process_noise(Z[:-1], dt, c.sigma_a, c.sigma_w) + np.diag(np.square(c.q_reg))
        L = np.linalg.cholesky(Q)
        out.append(np.linalg.solve(L, e[..., None])[..., 0].ravel())

        R = self.receivers(B)
        D = self.emitter(Z)[:, None, :] - R          # (K, nr, 3)
        rng = np.linalg.norm(D, axis=2)
        b = self.clock(g)
        if self.pinger:
            out.append((T[1:] - T[:-1] - c.Tp * (1 + s)) / c.sig_chain)
            pred = T[:, None] + rng / cs + b
            out.append(((m["toa"] - pred) / m["toa_sig"])[m["toa_mask"]])
        else:
            for q, (i, j) in enumerate(m["pairs"]):
                pred = (rng[:, i] - rng[:, j]) / cs + b[:, i] - b[:, j]
                out.append(((m["tdoa"][:, q] - pred) / m["tdoa_sig"][:, q])[m["tdoa_mask"][:, q]])

        # bearing at the shore array
        sh = self.geo.shore
        th = np.arctan2(Z[:, 1] - sh[1], Z[:, 0] - sh[0]) - alpha
        sb = m.get("bearing_sig", c.sig_bearing)          # per-epoch (K,) or scalar
        out.append((wrap(m["bearing"] - th) / sb)[m["bearing_mask"]])

        if self.has_mp:       # single-receiver horizontal range from multipath
            rh = np.hypot(D[..., 0], D[..., 1])
            out.append(((m["mp_range"] - rh) / m["mp_range_sig"])[m["mp_range_mask"]])
        if self.has_rr:       # range rate: line-of-sight unit vector . ground velocity
            V = Z[:, 3:4] * np.column_stack([np.cos(Z[:, 2]), np.sin(Z[:, 2])])
            if c.motion == "water":
                V = V + np.array([cx, cy])
            pr = (D[..., 0] * V[:, None, 0] + D[..., 1] * V[:, None, 1]) / rng
            out.append(((m["rr"] - pr) / m["rr_sig"])[m["rr_mask"]])

        # buoy drift + mooring prior
        out.append(((B[1:] - B[:-1]) / (c.q_buoy * np.sqrt(dt))).ravel())
        out.append(((B[0] - self.geo.buoys_nominal) / c.sig_buoy0).ravel())
        if c.start_prior is not None:
            x0, y0, s0 = c.start_prior
            out.append((Z[0, :2] - (x0, y0)) / s0)
        for k, x0, y0, s0 in c.pos_priors:
            out.append((Z[int(k), :2] - (x0, y0)) / s0)

        # global priors
        out.append((g - np.array(c.g_mean)) / np.array(c.g_sig))
        return np.concatenate(out)

    # ------------------------------------------------------- sparsity pattern
    def _build_sparsity(self):
        """One row per residual, in the same order as `residuals`. Each
        factor type adds an (nrows, width) array of the columns it touches;
        shorter rows are padded by repeating a column."""
        c, K, nb, m = self.cfg, self.K, self.cfg.nb, self.m
        iZ, iB, iG = self.iZ, self.iB, self.iG
        gb, ga, gsk, gc, gcx, gcy, gd = iG
        blocks, robust = [], []         # robust kind of each block's rows

        def add(b, rob):
            blocks.append(b); robust.append(rob)

        def col(x):                      # scalar index -> (K,) column
            return np.broadcast_to(x, (K,))

        def per_rx(timing):
            """(K, nr, 2) columns of each receiver: buoy position, or for the
            shore its clock (b_S, d_S) if timing else nothing (padded)."""
            out = np.empty((K, nb + 1, 2), int)
            out[:, :nb] = iB
            out[:, nb] = np.column_stack([col(gb), col(gd)]) if timing else iZ[:, :2]
            return out

        # motion
        mot = np.column_stack([iZ[:-1], iZ[1:], np.broadcast_to([gcx, gcy], (K - 1, 2))])
        add(np.repeat(mot, NX, axis=0), 1)
        if self.pinger:
            add(np.column_stack([self.iT[:-1], self.iT[1:], col(gsk)[:-1]]), 0)
            base = np.column_stack([iZ[:, :2], self.iT, col(gc)])          # (K, 4)
            cols = np.concatenate([np.repeat(base[:, None], nb + 1, 1), per_rx(True)], 2)
            add(cols[m["toa_mask"]], 2)
        else:
            rx = per_rx(True)
            base = np.column_stack([iZ[:, :2], col(gc)])
            for q, (i, j) in enumerate(m["pairs"]):
                add(np.column_stack([base, rx[:, i], rx[:, j]])[m["tdoa_mask"][:, q]], 2)
        add(np.column_stack([iZ[:, :2], col(ga)])[m["bearing_mask"]], 2)
        if self.has_mp:
            cols = np.concatenate([np.repeat(iZ[:, None, :2], nb + 1, 1), per_rx(False)], 2)
            add(cols[m["mp_range_mask"]], 2)
        if self.has_rr:
            base = np.column_stack([iZ[:, :4], col(gcx), col(gcy)])
            cols = np.concatenate([np.repeat(base[:, None], nb + 1, 1), per_rx(False)], 2)
            add(cols[m["rr_mask"]], 2)
        add(np.column_stack([iB[:-1].ravel(), iB[1:].ravel()]), 0)
        add(iB[0].reshape(-1, 1), 0)
        if c.start_prior is not None:
            add(iZ[0, :2].reshape(-1, 1), 0)
        for k, *_ in c.pos_priors:
            add(iZ[int(k), :2].reshape(-1, 1), 0)
        add(iG.reshape(-1, 1), 0)

        r, cidx = [], []
        o = 0
        for b in blocks:
            b = np.asarray(b, int)
            b = b.reshape(len(b), -1) if b.size else b.reshape(0, 1)   # empty block (all masked)
            r.append(np.repeat(o + np.arange(len(b)), b.shape[1])); cidx.append(b.ravel())
            o += len(b)
        r, cidx = np.concatenate(r), np.concatenate(cidx)
        S = coo_matrix((np.ones(r.size), (r, cidx)), shape=(o, self.n)).tocsc()
        S.data[:] = 1.0                  # duplicates (padding) were summed
        self.sparsity = S
        self.n_res = o
        # Row kind: 0 Gaussian (priors, ping chain, buoy drift: a robust prior
        # would only grow linearly beyond huber sigmas and could not hold e.g.
        # the sound speed), 1 motion (Huber: lets real manoeuvres through as
        # jumps), 2 measurement (Huber or Cauchy, see solve).
        self.robust = np.repeat(robust, [len(b) for b in blocks])

    # ------------------------------------------------------------------ solve
    def jacobian(self, v):
        """Forward differences with column grouping: columns that share no
        residual row are perturbed together (CPR), so one Jacobian costs one
        residual call per group (~16) instead of one per variable. Same step
        rule as scipy's 2-point scheme; values land directly in the CSC
        structure of the sparsity pattern."""
        S = self.sparsity
        if not hasattr(self, "_fd"):
            groups = group_columns(S)
            cols = np.repeat(np.arange(self.n), np.diff(S.indptr))
            self._fd = [(np.flatnonzero(groups == g), np.flatnonzero(groups[cols] == g))
                        for g in range(groups.max() + 1)]
            self._fd_cols = cols
        rows, cols = S.indices, self._fd_cols
        f0 = self.residuals(v)
        h = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(v))
        h = (v + h) - v                   # exactly representable steps
        data = np.empty(rows.size)
        for gcols, ent in self._fd:
            vp = v.copy(); vp[gcols] += h[gcols]
            df = self.residuals(vp) - f0
            data[ent] = df[rows[ent]] / h[cols[ent]]
        return csc_matrix((data, rows.copy(), S.indptr.copy()), shape=S.shape)

    def solve(self, v0, max_iter=60, verbose=False, fixed=(), loss=None):
        """Levenberg-Marquardt with a sparse direct solve of the normal
        equations (what GTSAM/Ceres do), robust loss via IRLS weights.

        loss (default cfg.loss):
          "huber"   Huber (scale cfg.huber sigmas) on measurement and motion rows.
          "cauchy"  Huber solve first, then continue with a redescending
                    Cauchy loss on the measurement rows (motion stays Huber).
                    Huber clips an outlier's pull to a constant, so one-sided
                    outliers (late multipath arrivals) still bias weakly
                    observable modes (buoy line + track translation) by tens
                    of metres; Cauchy lets their pull fade to zero.
        Priors, ping chain and buoy drift are always Gaussian.
        `fixed` lists variable indices held at their v0 value (for sweeps).
        Result: x, fun (whitened residuals), cost, nit, jac (IRLS-weighted)."""
        loss = loss or self.cfg.loss
        if loss not in ("huber", "cauchy"):
            raise ValueError(f"unknown loss {loss!r}")
        # Huber converges linearly (IRLS); as a warm start for Cauchy it only
        # needs to get past the gross initial errors. 1e-3 was too loose: on
        # the ForFun data it stopped on a plateau (cost 22e3 vs 3.3e3) and
        # Cauchy then settled in a different minimum.
        res = self._lm(v0, max_iter, verbose, fixed, cauchy=False,
                       tol=1e-4 if loss == "cauchy" else 1e-7)
        if loss == "cauchy":
            n1 = res.nit
            res = self._lm(res.x, max_iter, verbose, fixed, cauchy=True)
            res.nit += n1
        self.result = res
        return res

    def _robust(self, r, cauchy):
        """Robust cost and sqrt IRLS weights per row (see self.robust)."""
        k, kind = self.cfg.huber, self.robust
        a = np.abs(r)
        cau = (kind == 2) & cauchy
        hub = (kind > 0) & ~cau & (a > k)
        rho = np.where(hub, k * a - 0.5 * k**2, 0.5 * r**2)
        w = np.where(hub, k / np.maximum(a, 1e-12), 1.0)
        rho = np.where(cau, 0.5 * k**2 * np.log1p((r / k) ** 2), rho)
        w = np.where(cau, 1 / (1 + (r / k) ** 2), w)
        return rho.sum(), np.sqrt(w)

    def _lm(self, v0, max_iter, verbose, fixed, cauchy, tol=1e-7):
        v = v0.copy()
        free = np.setdiff1d(np.arange(self.n), np.asarray(fixed, dtype=int))
        r = self.residuals(v)
        cost, sw = self._robust(r, cauchy)
        lam = 1e-3
        for it in range(max_iter):
            J = self.jacobian(v)[:, free]
            Jw = diags(sw) @ J
            H = (Jw.T @ Jw).tocsc()
            gvec = Jw.T @ (sw * r)
            d = H.diagonal() + 1e-12
            while True:
                A = (H + diags(lam * d)).tocsc()
                dx = -spd_lu(A).solve(gvec)
                vn = v.copy(); vn[free] += dx
                rn = self.residuals(vn)
                cn, swn = self._robust(rn, cauchy)
                if cn < cost:
                    break
                lam *= 10
                if lam > 1e10:
                    break
            if verbose:
                print(f"  it {it:2d} cost {cost:12.2f} -> {cn:12.2f}  lam {lam:.1e}"
                      f"{'  (cauchy)' if cauchy else ''}")
            if cn >= cost:
                break
            rel = (cost - cn) / cost
            v, r, cost, sw = vn, rn, cn, swn
            lam = max(lam / 10, 1e-9)
            if rel < tol:
                break
        self.free = free
        return SimpleNamespace(x=v, fun=r, cost=cost, nit=it + 1,
                               jac=(diags(sw) @ self.jacobian(v)[:, free]).tocsc())

    def marginal_cov(self, res, idx):
        """Covariance block of selected (free) variables, from J^T J."""
        H = (csc_matrix(res.jac).T @ csc_matrix(res.jac)).tocsc()
        pos = np.searchsorted(self.free, idx)
        E = np.zeros((len(self.free), len(idx)))
        E[pos, np.arange(len(idx))] = 1.0
        return spd_lu(H).solve(E)[pos]

    def marginal_std(self, res, idx):
        """Std of selected variables from the Gauss-Newton information J^T J."""
        J = csc_matrix(res.jac)
        H = (J.T @ J).tocsc()
        lu = spd_lu(H)
        pos = np.searchsorted(self.free, idx)   # idx must be free variables
        E = np.zeros((len(self.free), len(idx)))
        E[pos, np.arange(len(idx))] = 1.0
        X = lu.solve(E)
        return np.sqrt(np.abs(X[pos, np.arange(len(idx))]))

    def conditional_std(self, res, idx, given):
        """Std of selected variables with the variables `given` held fixed (the
        Gauss-Newton information J^T J restricted to the other free variables)."""
        keep = self.free[~np.isin(self.free, given)]
        J = csc_matrix(res.jac)[:, np.searchsorted(self.free, keep)]
        lu = spd_lu((J.T @ J).tocsc())
        pos = np.searchsorted(keep, idx)
        E = np.zeros((len(keep), len(idx)))
        E[pos, np.arange(len(idx))] = 1.0
        return np.sqrt(np.abs(lu.solve(E)[pos, np.arange(len(idx))]))


# ---------------------------------------------------------------- initialise
def _soft_l1(z):
    return 2 * (np.sqrt(1 + z**2) - 1)


def _running_nanmedian(x, w):
    pad = np.pad(x, w // 2, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows
        return np.nanmedian(win, axis=1)


def initial_guess(fg: FactorGraph):
    """Single-epoch fixes (nominal buoys, prior globals), then smoothed
    finite-difference course/speed. Good enough for the batch solver.

    Each fix is a small robust least squares from the previous fix. A
    coarse grid search over the region (cfg.region, cfg.grid_step, with
    every sigma inflated to the grid step) seeds the first fix, re-seeds
    after a dropout longer than cfg.reseed_gap epochs, and replaces any
    local fix that fits badly or jumps further than u_max allows. The grid
    cost includes a soft pull towards cfg.start_prior (first fix) or the
    last fix (sigma = reachable distance), which resolves the mirror
    ambiguity when only two constraints are present.

    Phase 1 runs twice: pass 1 profiles out each ping's emission time, a
    robust line through those times gives the ping chain (and the initial
    skew); pass 2 re-fixes with the chain times, so a single TOA is a range.
    The raw pass-2 (or phase 2) fixes are kept in fg.init_fixes."""
    c, geo, K, m = fg.cfg, fg.geo, fg.K, fg.m
    nb, nr = c.nb, fg.nr
    gm, gs = np.array(c.g_mean), np.array(c.g_sig)
    cs = gm[3]
    R = np.empty((nr, 3))
    R[:nb, :2] = geo.buoys_nominal; R[:nb, 2] = geo.buoy_depth; R[nb] = geo.shore
    bk = np.zeros((K, nr)); bk[:, nb] = gm[0] + gm[6] * fg.tk
    sh = geo.shore
    sig_brg = np.hypot(c.sig_bearing, gs[1])     # alpha is unknown to its prior sigma
    if c.region is None:
        lo, hi = R[:, :2].min(0) - 1500.0, R[:, :2].max(0) + 1500.0
        region = (lo[0], hi[0], lo[1], hi[1])
    else:
        region = c.region
    gx = np.arange(region[0], region[1] + 1e-9, c.grid_step)
    gy = np.arange(region[2], region[3] + 1e-9, c.grid_step)
    GRID = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)

    def res(P, k, Tk=None, floor=0.0):
        """Whitened residuals (N, m) of epoch k at candidate positions P (N, 2),
        and the profiled emission time (N,). floor [m] inflates all sigmas."""
        P = np.atleast_2d(P)
        N = len(P)
        D = np.column_stack([P, np.full(N, geo.src_depth)])[:, None, :] - R[None]
        rr = np.linalg.norm(D, axis=2)                   # (N, nr)
        z, t = [], np.full(N, np.nan)
        if fg.pinger:
            ok = m["toa_mask"][k]
            if ok.any():
                s = np.hypot(m["toa_sig"][k, ok], floor / cs)
                a = m["toa"][k, ok] - rr[:, ok] / cs - bk[k, ok]    # emission time per rx
                t = (a / s**2).sum(1) / (1 / s**2).sum() if Tk is None else np.full(N, Tk)
                z.append((a - t[:, None]) / s)
        else:
            for q, (i, j) in enumerate(m["pairs"]):
                if m["tdoa_mask"][k, q]:
                    s = np.hypot(m["tdoa_sig"][k, q], floor / cs)
                    pr = (rr[:, i] - rr[:, j]) / cs + bk[k, i] - bk[k, j]
                    z.append(((m["tdoa"][k, q] - pr) / s)[:, None])
        if m["bearing_mask"][k]:
            dist = np.maximum(np.hypot(P[:, 0] - sh[0], P[:, 1] - sh[1]), 1.0)
            th = np.arctan2(P[:, 1] - sh[1], P[:, 0] - sh[0]) - gm[1]
            z.append((wrap(m["bearing"][k] - th) / np.hypot(sig_brg, floor / dist))[:, None])
        if fg.has_mp and m["mp_range_mask"][k].any():
            ok = m["mp_range_mask"][k]
            rh = np.hypot(D[..., 0], D[..., 1])[:, ok]
            z.append((m["mp_range"][k, ok] - rh) / np.hypot(m["mp_range_sig"][k, ok], floor))
        return (np.column_stack(z) if z else np.zeros((N, 0))), t

    def n_info(k, timed):
        """Number of independent position constraints at epoch k."""
        if fg.pinger:
            nt = m["toa_mask"][k].sum()
            n = nt if timed else max(nt - 1, 0)
        else:
            q = np.flatnonzero(m["tdoa_mask"][k])
            rx = {r for p in q for r in m["pairs"][p]}
            n = min(len(q), max(len(rx) - 1, 0))
        n += int(m["bearing_mask"][k])
        if fg.has_mp:
            n += m["mp_range_mask"][k].sum()
        return n

    def local(k, u0, Tk):
        sol = least_squares(lambda u: res(u, k, Tk)[0][0], u0, x_scale=100.0,
                            loss="soft_l1", f_scale=3.0)
        t = res(sol.x, k, Tk)[1][0]
        zq = res(sol.x, k, Tk, floor=c.grid_step)[0][0]   # misfit in ~grid-step units
        return sol.x, t, zq

    def grid(k, Tk, centre, sig_c):
        z, _ = res(GRID, k, Tk, floor=c.grid_step)
        cost = _soft_l1(z / 3).sum(1)
        if centre is not None:
            cost = cost + np.sum((GRID - centre) ** 2, 1) / sig_c**2
        return local(k, GRID[np.argmin(cost)], Tk)

    def track(Tch=None, seeds=None):
        """One pass of fixes over all epochs. Returns XY (K, 2) (NaN where
        no fix) and the profiled emission times (K,)."""
        XY = np.full((K, 2), np.nan); T0 = np.full(K, np.nan)
        prev, kprev = None, None
        for k in range(K):
            Tk = None if Tch is None else Tch[k]
            if n_info(k, Tk is not None) < 2:
                continue
            reach = None if prev is None else c.u_max * (k - kprev) * c.Tp + 3 * c.grid_step
            if prev is None:
                sp = c.start_prior
                u, t, z = grid(k, Tk, None if sp is None else np.array(sp[:2]),
                               None if sp is None else np.hypot(sp[2], c.grid_step))
            elif k - kprev > c.reseed_gap:
                u, t, z = grid(k, Tk, prev, reach)
            else:
                u0 = seeds[k] if seeds is not None and np.isfinite(seeds[k]).all() else prev
                u, t, z = local(k, u0, Tk)
                if np.median(np.abs(z)) > 3 or np.hypot(*(u - prev)) > reach:
                    u2, t2, z2 = grid(k, Tk, prev, reach)
                    if np.hypot(*(u2 - prev)) <= reach or np.median(np.abs(z2)) < np.median(np.abs(z)):
                        u, t, z = u2, t2, z2
            if not (region[0] <= u[0] <= region[1] and region[2] <= u[1] <= region[3]):
                continue                  # diverged (poor geometry): no fix
            XY[k], T0[k] = u, t
            prev, kprev = u, k
        return XY, T0

    ks = np.arange(K)
    skew0 = gm[2]
    if fg.pinger:
        XY1, T0 = track()
        good = np.isfinite(T0)
        if good.sum() < 2:
            raise ValueError("initial_guess: fewer than 2 epochs with a fix")
        # robust line T0 = a + b k  -> ping chain and skew
        for _ in range(3):
            A = np.column_stack([np.ones(good.sum()), ks[good]])
            a, b = np.linalg.lstsq(A, T0[good], rcond=None)[0]
            e = T0 - (a + b * ks)
            mad = np.nanmedian(np.abs(e[good])) + 1e-9
            good = np.isfinite(T0) & (np.abs(e) < 5 * 1.4826 * mad + 2 * c.sig_toa)
        # The slope is biased by systematic fix errors (buoys, c), so keep the
        # skew within 2 prior sigmas and refit the intercept with it.
        skew0 = np.clip(b / c.Tp - 1, gm[2] - 2 * gs[2], gm[2] + 2 * gs[2])
        tch = np.median((T0 - ks * c.Tp * (1 + skew0))[good]) + ks * c.Tp * (1 + skew0)
        XY, _ = track(tch, seeds=XY1)
    else:
        XY, _ = track()
    fg.init_fixes = XY.copy()

    # reject jumps against a running median, they are outlier-driven fixes
    med = np.column_stack([_running_nanmedian(XY[:, d], 9) for d in range(2)])
    XY[np.hypot(*(XY - med).T) > 30] = np.nan

    # fill gaps, smooth, get course/speed
    good = np.isfinite(XY[:, 0])
    if good.sum() < 2:
        raise ValueError("initial_guess: fewer than 2 usable fixes")
    for d in range(2):
        XY[:, d] = np.interp(ks, ks[good], XY[good, d])
    w = 5
    ker = np.ones(w) / w
    XYs = np.column_stack([np.convolve(np.pad(XY[:, d], w // 2, mode="edge"), ker, "valid") for d in range(2)])
    V = np.gradient(XYs, c.Tp, axis=0)
    if c.motion == "water":   # remove the prior current from the ground velocity
        V = V - gm[4:6]
    Z = np.column_stack([XY, np.unwrap(np.arctan2(V[:, 1], V[:, 0])), np.hypot(*V.T), np.zeros(K)])

    v = np.zeros(fg.n)
    v[fg.iZ] = Z
    if fg.pinger:
        v[fg.iT] = tch
    v[fg.iB] = np.broadcast_to(geo.buoys_nominal, (K, nb, 2))
    v[fg.iG] = gm
    if fg.pinger:
        v[fg.iG[2]] = skew0
    return v


def ground_velocity(fg: FactorGraph, v):
    """Emitter velocity over ground [vx, vy] per epoch."""
    Z, _, _, g = fg.unpack(v)
    V = np.column_stack([Z[:, 3] * np.cos(Z[:, 2]), Z[:, 3] * np.sin(Z[:, 2])])
    return V + g[4:6] if fg.cfg.motion == "water" else V
