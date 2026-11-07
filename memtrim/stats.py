"""Statistical tests used in §2.

* ``two_lines``  - Simonsohn (2018) two-lines test for a U / inverted-U
  relationship: interrupted regression with the breakpoint chosen by the
  "Robin Hood" procedure; the shape is supported when both slopes have
  opposite signs and are each significant.
* ``quadratic``  - OLS with a quadratic term (reported alongside, as in §2).
* ``overlap_tests`` - RQ2: compares overlap of successful vs failed cases.
  Mann-Whitney U on raw overlap, a one-sided test that failures lie closer
  to the centre of the range, and the two-lines test of P(failure | overlap)
  for an inverted U.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC3 robust standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (resid / np.clip(1 - h, 1e-8, None)) ** 2
    cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def _interrupted(x: np.ndarray, y: np.ndarray, xc: float) -> dict:
    high = (x >= xc).astype(float)
    xlow = np.where(x < xc, x - xc, 0.0)
    xhigh = np.where(x >= xc, x - xc, 0.0)
    X = np.column_stack([np.ones_like(x), xlow, xhigh, high])
    beta, se = _ols(X, y)
    df = max(len(x) - X.shape[1], 1)
    z = beta / np.where(se > 0, se, np.inf)
    p = 2 * sps.t.sf(np.abs(z), df)
    return dict(b1=beta[1], b2=beta[2], z1=z[1], z2=z[2], p1=p[1], p2=p[2])


def _smooth(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-polynomial smoother with pointwise SEs (stand-in for the GAM in
    the original R code; adequate for a single bump)."""
    X = np.column_stack([x**p for p in range(4)])
    beta, _ = _ols(X, y)
    resid = y - X @ beta
    sigma2 = resid @ resid / max(len(y) - 4, 1)
    G = np.column_stack([grid**p for p in range(4)])
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return G @ beta, np.sqrt(np.einsum("ij,jk,ik->i", G, cov, G))


def two_lines(x, y, shape: str = "u") -> dict:
    """shape='u' tests decrease-then-increase; 'inverted' the opposite."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(np.unique(x)) < 4:
        return dict(ok=False, reason="too few distinct x")
    grid = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 50)))
    fit, se = _smooth(x, y, grid)
    i_ext = int(np.argmin(fit) if shape == "u" else np.argmax(fit))
    ext, ext_se = fit[i_ext], se[i_ext]
    flat = grid[(fit <= ext + ext_se) if shape == "u" else (fit >= ext - ext_se)]
    first = _interrupted(x, y, grid[i_ext])
    z1, z2 = abs(first["z1"]), abs(first["z2"])
    q = z2 / (z1 + z2) if (z1 + z2) > 0 else 0.5
    xc = float(np.quantile(flat, q)) if len(flat) else float(grid[i_ext])
    res = _interrupted(x, y, xc)
    expect = (-1, 1) if shape == "u" else (1, -1)
    ok = (np.sign(res["b1"]) == expect[0] and np.sign(res["b2"]) == expect[1]
          and res["p1"] < 0.05 and res["p2"] < 0.05)
    return dict(ok=bool(ok), breakpoint=xc, slope1=float(res["b1"]), slope2=float(res["b2"]),
                p1=float(res["p1"]), p2=float(res["p2"]), shape=shape)


def quadratic(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x, x**2])
    beta, se = _ols(X, y)
    df = max(len(x) - 3, 1)
    t = beta[2] / se[2] if se[2] > 0 else np.inf
    return dict(b2=float(beta[2]), p=float(2 * sps.t.sf(abs(t), df)))


def overlap_tests(success: list[float], failure: list[float]) -> dict:
    s, f = np.asarray(success, float), np.asarray(failure, float)
    out: dict = dict(n_success=len(s), n_failure=len(f))
    if len(s) < 3 or len(f) < 3:
        out["ok"] = False
        return out
    out["mwu_p"] = float(sps.mannwhitneyu(s, f, alternative="two-sided").pvalue)
    centre = float(np.median(np.concatenate([s, f])))
    out["centre"] = centre
    out["centrality_p"] = float(sps.mannwhitneyu(np.abs(f - centre), np.abs(s - centre), alternative="less").pvalue)
    x = np.concatenate([s, f])
    yfail = np.concatenate([np.zeros_like(s), np.ones_like(f)])
    out["two_lines"] = two_lines(x, yfail, shape="inverted")
    out["ok"] = True
    return out


def mean_ci(values, conf: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), 0.0
    return float(v.mean()), float(sps.t.ppf((1 + conf) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
