"""
Maximum-likelihood estimation of multivariate Hawkes parameters from event data.

Fits: mu (4,), alpha (4x4), beta (4,)
using the log-likelihood gradient via scipy.optimize.minimize.

Reference: Bacry, Mastromatteo, Muzy (2015) — "Hawkes processes in finance."
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from typing import NamedTuple

from ..sim.hawkes import MultivariateHawkes, N_TYPES


def _log_likelihood(params_flat: np.ndarray, events, T: float) -> float:
    """
    Negative log-likelihood of multivariate Hawkes given events.

    params_flat: [mu (4), alpha (16), beta (4)] = 24-dim vector
    Constraint: mu > 0, alpha > 0, beta > 0 (enforced via bounds in optimizer).
    """
    mu = params_flat[:N_TYPES]
    alpha = params_flat[N_TYPES: N_TYPES + N_TYPES * N_TYPES].reshape(N_TYPES, N_TYPES)
    beta = params_flat[N_TYPES + N_TYPES * N_TYPES:]

    if np.any(mu <= 0) or np.any(beta <= 0):
        return 1e10

    times = np.array([e.time for e in events])
    etypes = np.array([e.etype for e in events])
    n = len(events)

    # Running intensity at each event time (O(n^2) — ok for calibration)
    log_lam_sum = 0.0
    for i in range(n):
        t_i = times[i]
        m = etypes[i]
        # Intensity at t_i
        R = mu[m]
        for j in range(i):
            R += alpha[m, etypes[j]] * np.exp(-beta[m] * (t_i - times[j]))
        R = max(R, 1e-10)
        log_lam_sum += np.log(R)

    # Integral term: sum_m [ mu_m * T + sum_i alpha[m,type_i]/beta[m] * (1 - exp(-beta[m]*(T-t_i))) ]
    integral = 0.0
    for m in range(N_TYPES):
        integral += mu[m] * T
        for i in range(n):
            integral += alpha[m, etypes[i]] / beta[m] * (1.0 - np.exp(-beta[m] * (T - times[i])))

    nll = -(log_lam_sum - integral)
    return float(nll)


def fit_hawkes(
    events,
    T: float,
    n_restarts: int = 3,
    seed: int = 0,
    verbose: bool = True,
) -> MultivariateHawkes:
    """
    Fit Hawkes parameters via MLE.

    events : list of Event(time, etype) — from lobster_loader or hawkes.simulate()
    T      : observation window length (seconds)

    Returns fitted MultivariateHawkes.
    """
    rng = np.random.default_rng(seed)
    best_result = None
    best_nll = np.inf

    n_params = N_TYPES + N_TYPES * N_TYPES + N_TYPES  # 24

    for restart in range(n_restarts):
        # Random init
        mu0 = rng.uniform(0.5, 5.0, N_TYPES)
        alpha0 = rng.uniform(0.01, 0.3, (N_TYPES, N_TYPES))
        beta0 = rng.uniform(2.0, 8.0, N_TYPES)
        x0 = np.concatenate([mu0, alpha0.flatten(), beta0])

        bounds = [(1e-4, None)] * n_params

        result = minimize(
            _log_likelihood,
            x0,
            args=(events, T),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result

        if verbose:
            print(f"  Restart {restart+1}/{n_restarts}: NLL={result.fun:.2f}  success={result.success}")

    x = best_result.x
    mu = x[:N_TYPES]
    alpha = x[N_TYPES: N_TYPES + N_TYPES * N_TYPES].reshape(N_TYPES, N_TYPES)
    beta = x[N_TYPES + N_TYPES * N_TYPES:]

    fitted = MultivariateHawkes(mu, alpha, beta)
    if verbose:
        print(f"Fitted Hawkes. Branching ratio: {fitted.branching_ratio():.3f} (< 1 = stationary)")
    return fitted


def save_params(hawkes: MultivariateHawkes, path: str | Path) -> None:
    """Save fitted params to JSON."""
    params = {
        "mu": hawkes.mu.tolist(),
        "alpha": hawkes.alpha.tolist(),
        "beta": hawkes.beta.tolist(),
        "branching_ratio": hawkes.branching_ratio(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved calibrated params to {path}")


def load_params(path: str | Path) -> MultivariateHawkes:
    """Load fitted params from JSON."""
    with open(path) as f:
        d = json.load(f)
    return MultivariateHawkes(
        mu=np.array(d["mu"]),
        alpha=np.array(d["alpha"]),
        beta=np.array(d["beta"]),
    )
