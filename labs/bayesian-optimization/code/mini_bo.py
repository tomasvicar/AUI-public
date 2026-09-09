"""`mini_bo` - Bayesian optimization in fifty lines.

This is the **reference solution** of the notebook `notebooks/ex1_bo.ipynb`.
In class the students write the same thing, with three places hollowed out
(both acquisition functions, `suggest()`, and the random-search baseline).

There is no class and no library interface - just three functions on top of
a Gaussian process:

    suggest(x, y, rng)                  where to measure next
    bayesian_optimization(f, ...)       the whole loop: random start, then suggest
    best_from_model(x, y)               what to report at the end

The point of the lab: between "I call a library" and "I wrote it myself" there
is no chasm, only fifty lines - and whoever wrote them once can tell when the
library is misbehaving.

The Gaussian process has an RBF kernel with a **fixed** length scale per axis
(`KERNEL_LENGTH`, in mg/m2 and in days), a fixed amplitude (`AMPLITUDE`) and
knows the measurement noise (`SIGMA_NOISE`). The lecture estimates the kernel
from data; here it is fixed, and part 2 of the notebook points the same code at
a classifier by reassigning these constants and the bounds - nothing else.

Run one example optimization against the patient simulator:

    uv run python labs/bayesian-optimization/code/mini_bo.py
"""

import numpy as np
from scipy.stats import norm

from black_box import BOUNDS, SIGMA_NOISE, random_points

KERNEL_LENGTH = np.array([30.4, 9.5])   # [mg/m2, days] - one per axis, they are not comparable
AMPLITUDE = 43.0                        # sd of the prior [score points]
KAPPA = 2.0                             # exploration/exploitation trade-off of UCB


# --- Gaussian process ------------------------------------------------------


def rbf_kernel(a, b):
    """k(x, x') = A^2 exp(-sum_i (x_i - x'_i)^2 / (2 l_i^2))"""
    d2 = (((a[:, None, :] - b[None, :, :]) / KERNEL_LENGTH) ** 2).sum(-1)
    return AMPLITUDE**2 * np.exp(-0.5 * d2)


def gp_posterior(x_data, y_data, x_query):
    """Posterior mean and standard deviation at the points x_query."""
    mean = y_data.mean()
    kk = rbf_kernel(x_data, x_data) + (SIGMA_NOISE**2 + 1e-8) * np.eye(len(x_data))
    ks = rbf_kernel(x_data, x_query)
    alpha = np.linalg.solve(kk, y_data - mean)
    v = np.linalg.solve(kk, ks)

    mu = mean + ks.T @ alpha
    variance = AMPLITUDE**2 - (ks * v).sum(0)
    return mu, np.sqrt(np.clip(variance, 1e-12, None))


# --- Acquisition functions -------------------------------------------------


def ei(mu, sigma, best):
    """Expected improvement over the best value measured so far."""
    diff = mu - best
    z = diff / sigma
    return diff * norm.cdf(z) + sigma * norm.pdf(z)


def ucb(mu, sigma, best=None):
    """Upper confidence bound - optimism in the face of uncertainty."""
    return mu + KAPPA * sigma


# --- The loop --------------------------------------------------------------


def suggest(x, y, rng, acquisition=ei):
    """Where to measure next: the acquisition's argmax over 2000 random candidates."""
    candidates = random_points(rng, 2000)
    mu, sigma = gp_posterior(x, y, candidates)
    return candidates[np.argmax(acquisition(mu, sigma, y.max()))]


def bayesian_optimization(f, n_random=5, n_steps=15, acquisition=ei, seed=0):
    """`n_random` random points, then `n_steps` times: suggest, measure, add.

    `f(d, T)` is one expensive measurement; the budget is `n_random + n_steps`.
    Returns the measured points (n, 2) and their values (n,).
    """
    rng = np.random.default_rng(seed)
    x = random_points(rng, n_random)
    y = np.array([f(d, T) for d, T in x])
    for _ in range(n_steps):
        point = suggest(x, y, rng, acquisition)
        x, y = np.vstack([x, point]), np.append(y, f(*point))
    return x, y


def best_from_model(x, y):
    """The point where the model's MEAN is largest - what to report.

    The best MEASURED point, `x[np.argmax(y)]`, is inflated by favourable noise.
    """
    dd, tt = np.linspace(*BOUNDS[0], 201), np.linspace(*BOUNDS[1], 201)
    grid = np.array([(d, T) for d in dd for T in tt])
    mu, _ = gp_posterior(x, y, grid)
    return grid[np.argmax(mu)]


# --- Example ---------------------------------------------------------------


def main():
    import black_box as cs

    rng = np.random.default_rng(0)
    x, y = bayesian_optimization(lambda d, T: float(cs.measure(d, T, rng)))
    measured = x[np.argmax(y)]
    recommended = best_from_model(x, y)
    mu, _ = gp_posterior(x, y, np.array([recommended]))

    print("mini_bo on the patient simulator (budget of 20 evaluations)")
    print("-" * 58)
    print(f"best measurement:  {y.max():.2f} points "
          f"at (d, T) = ({measured[0]:.1f}, {measured[1]:.1f})")
    print(f"model says:        {mu[0]:.2f} points "
          f"at (d, T) = ({recommended[0]:.1f}, {recommended[1]:.1f})")
    print(f"truth:             {cs.SCORE_OPT:.2f} points "
          f"at ({cs.DOSE_OPT:.0f}, {cs.INTERVAL_OPT:.0f})")
    print(f"true score where the best measurement points: "
          f"{float(cs.score(*measured)):.2f}, where the model points: "
          f"{float(cs.score(*recommended)):.2f}")


if __name__ == "__main__":
    main()
