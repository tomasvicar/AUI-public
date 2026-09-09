"""Bayesian optimization from scratch - the whole method on one screen.

The script searches for the chemotherapy dosing regimen used throughout the
lecture: the dose per cycle `d` and the interval between cycles `T`. The
patient simulator (`patient_simulator.py`, function `measure` in
`common.py`) plays the role of an **expensive noisy black box** - it returns
one number and nothing else.

Where things come from:

* the **Gaussian process** is taken from `scikit-learn`
  (`GaussianProcessRegressor`), because that is how it is done in practice:
  the kernel is assembled from ready-made pieces and the library estimates its
  hyperparameters itself by maximizing the marginal likelihood;
* the **acquisition functions PI, EI and UCB are written out by hand here** -
  three formulas, two lines each, and they show that there is nothing behind
  them but a substitution into the normal distribution function;
* the **loop** is the five steps from the slide: fit the model, maximize the
  acquisition, one expensive evaluation, add the data, and repeat.

**A warning about the numbers.** The figures in the lecture are drawn by the
Gaussian process written in NumPy in `common.py`, and that one has the
kernel length scale and amplitude **fixed** (0.38 and 43.0). This script lets
`scikit-learn` estimate them from the data instead, so the model is slightly
different at every step and **the run need not match the animation
`bo-loop.gif`** or the reference run in the slides down to the last decimal.
That is not a bug in either of them: it is the same method in two
implementations, and the difference is exactly the "choice of kernel" the
lecture talks about. What **should** match is the conclusion.

And it does. With seed 4 the script prints this:

* **EI** recommends $d$ = 62.7 mg/m2, $T$ = 22.1 days, true score **41.96**,
  that is **0.18 points below the optimum of 42.13** after twelve evaluations;
* **UCB** recommends 61.3 and 22.5 with a score of 41.84 (a loss of 0.29);
* **PI** rushes it - it crawls into the first decent hill, ends at 38.42, and
  makes a nice illustration of why PI is not used much in practice.

The figure `figures/bo-from-scratch.svg` is the output of **this script** for
whoever runs it; the slides do not use it (they show the animation
`bo-loop.gif`).

Run it with::

    uv run python lectures/bayesian-optimization/code/bo_from_scratch.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import common as s

# On twelve points the estimated length scale sometimes lands on the edge of
# the allowed range and scikit-learn warns about it. That is normal and
# harmless here - it would only flood the printout of the run we care about.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_START = 4      # points measured at random before any model is built
N_STEPS = 8      # steps of the loop; twelve evaluations in total
SEED = 4
KAPPA = 2.0      # exploration/exploitation trade-off of the UCB acquisition


# --- acquisition functions: three formulas, nothing more -------------------
#
# All three get the prediction of the model (mu, sigma) and the best value
# measured so far, f+; they return a number saying "how much it is worth
# asking here". It gets maximized.

def probability_of_improvement(mu, sigma, best):
    """PI: the chance that it will be better here than anything so far."""
    return norm.cdf((mu - best) / sigma)


def expected_improvement(mu, sigma, best):
    """EI: by how much it will be better - on average, over all improvements."""
    z = (mu - best) / sigma
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)


def upper_confidence_bound(mu, sigma, best=None, kappa=KAPPA):
    """UCB: optimism in the face of uncertainty."""
    return mu + kappa * sigma


ACQUISITIONS = {"PI": probability_of_improvement,
                "EI": expected_improvement,
                "UCB": upper_confidence_bound}


# --- the Gaussian process from scikit-learn --------------------------------

def build_model(points, values):
    """Fits a Gaussian process to the points measured so far.

    The kernel is a product of a constant (the amplitude) and a Gaussian
    kernel (the length scale) plus white noise, that is exactly the three
    hyperparameters from the lecture. The constructor only gets an initial
    guess; `fit` then tunes them by maximizing the marginal likelihood - hence
    `n_restarts_optimizer`, that task has local maxima of its own.

    Two things are done by hand so that they stay visible: the coordinates are
    rescaled to the unit cube (mg/m2 and days are not comparable units and a
    single length scale would have to serve both) and the mean of the data is
    subtracted (a Gaussian process is drawn towards zero a priori). Thanks to
    that the hyperparameters stay in **score points** and can be compared with
    the numbers in the lecture. Returns the pair (model, subtracted mean).
    """
    # We know the noise of the patient response from the simulator (2 score
    # points), so we simply tell the model - hence `"fixed"`. If we did not
    # know it, it would be estimated along with the rest, and on twelve points
    # that comes out rather wild: the model happily explains the noise away as
    # real structure of the function and becomes overconfident. That is exactly
    # the underestimated uncertainty from the slides.
    kernel = (ConstantKernel(43.0 ** 2, (1e0, 1e6))
              * RBF(0.4, (5e-2, 5e0))
              + WhiteKernel(s.SIGMA_NOISE ** 2, "fixed"))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=False, random_state=0)
    mean = float(np.mean(values))
    return model.fit(s.to_unit_cube(points), values - mean), mean


def predict(model, mean, where):
    """Prediction of the model in physical coordinates: (mu, sigma)."""
    mu, sigma = model.predict(s.to_unit_cube(where), return_std=True)
    return mu + mean, sigma


def optimize(acquisition_name="EI", n_steps=N_STEPS, seed=SEED,
            verbose=True):
    """The whole Bayesian optimization loop. Returns the history of the run."""
    rng = np.random.default_rng(seed)
    _, _, candidates = s.candidate_grid()      # 61 x 61 points we may ask about
    acquisition = ACQUISITIONS[acquisition_name]

    # 1) initial points: spread out evenly, still without any model
    points = np.column_stack([
        rng.permutation(np.linspace(*s.BOUNDS[0], N_START + 2)[1:-1]),
        rng.permutation(np.linspace(*s.BOUNDS[1], N_START + 2)[1:-1]),
    ])
    values = s.measure(points[:, 0], points[:, 1], rng)

    if verbose:
        print(f"\n{acquisition_name} acquisition: {N_START} initial points "
              f"+ {n_steps} steps")
        for point, value in zip(points, values):
            print(f"  start  : d = {point[0]:6.1f}, T = {point[1]:5.1f}  ->  "
                  f"measured {value:8.2f}")

    for step in range(1, n_steps + 1):
        # 2) fit a model of the data collected so far
        model, mean = build_model(points, values)
        mu, sigma = predict(model, mean, candidates)

        # 3) maximize the acquisition (on a grid - it is cheap, we can afford it)
        a = acquisition(mu, sigma, values.max())
        new_point = candidates[int(np.argmax(a))]

        # 4) one expensive evaluation and 5) adding the data
        measured = float(s.measure(new_point[0], new_point[1], rng))
        points = np.vstack([points, new_point])
        values = np.append(values, measured)

        if verbose:
            print(f"  step {step}: d = {new_point[0]:6.1f}, T = {new_point[1]:5.1f}  ->  "
                  f"measured {measured:8.2f}   (max acquisition {a.max():7.2f}, "
                  f"model sigma {sigma[int(np.argmax(a))]:5.2f})")

    # at the end we recommend the maximum of the model, not the best measurement
    model, mean = build_model(points, values)
    mu, sigma = predict(model, mean, candidates)
    recommendation = candidates[int(np.argmax(mu))]
    return {"points": points, "values": values, "model": model, "mu": mu,
            "sigma": sigma, "candidates": candidates, "recommendation": recommendation}


def summarize(name, run):
    """Prints what the run recommends - and how far that is from the optimum."""
    best = int(np.argmax(run["values"]))
    best_point = run["points"][best]
    rec = run["recommendation"]
    print(f"\n{name}: best MEASURED value {run['values'][best]:.2f} "
          f"at d = {best_point[0]:.1f}, T = {best_point[1]:.1f} "
          f"(without noise {float(s.score(*best_point)):.2f})")
    print(f"{' ' * len(name)}  MODEL RECOMMENDS d = {rec[0]:.1f}, "
          f"T = {rec[1]:.1f}, true score {float(s.score(*rec)):.2f} "
          f"(optimum {s.SCORE_OPT:.2f}, loss "
          f"{s.SCORE_OPT - float(s.score(*rec)):.2f})")
    print(f"{' ' * len(name)}  estimated kernel: {run['model'].kernel_}")


# --- figure: what the model knows after twelve evaluations ------------------

def draw(run, name="bo-from-scratch"):
    """Saves three panels: the truth, the model mean and its uncertainty."""
    d, t, candidates = s.candidate_grid()
    dd, tt = np.meshgrid(d, t, indexing="ij")
    shape = (len(d), len(t))
    d_fine, t_fine, s_fine = s.landscape(241)
    ddf, ttf = np.meshgrid(d_fine, t_fine, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True,
                            layout="constrained")
    panels = (
        (s_fine, ddf, ttf, s.LEVELS, "Blues",
         "true landscape (the algorithm never sees this)"),
        (run["mu"].reshape(shape), dd, tt, s.LEVELS, "GnBu",
         "model: mean mu"),
        (run["sigma"].reshape(shape), dd, tt, np.linspace(0, 44, 12), "BuGn",
         "model: uncertainty sigma"),
    )
    for ax, (z, x, y, levels, cmap, title) in zip(axes, panels):
        ax.contourf(x, y, z, levels=levels, cmap=cmap, extend="both")
        ax.contour(x, y, z, levels=levels, colors="#7f929e", linewidths=0.5)
        ax.scatter(run["points"][:, 0], run["points"][:, 1], s=40,
                    facecolor="white", edgecolor=s.BLUE, linewidth=1.4,
                    clip_on=False, zorder=6)
        ax.scatter([s.DOSE_OPT], [s.INTERVAL_OPT], s=130, marker="X",
                    color=s.RED, edgecolor="white", zorder=7)
        ax.scatter([run["recommendation"][0]], [run["recommendation"][1]], s=190,
                    marker="*", color=s.ORANGE, edgecolor="white", zorder=8)
        ax.set(xlim=s.BOUNDS[0], ylim=s.BOUNDS[1], title=title)
        ax.set_xlabel("dose per cycle d [mg/m2]")
    axes[0].set_ylabel("interval T [days]")
    fig.suptitle("bo_from_scratch.py: 12 evaluations, X = optimum, "
                 "star = recommendation of the model", color=s.BLUE)
    s.save_figure(fig, name)


def main():
    run = optimize("EI")
    summarize("EI", run)
    draw(run)

    # The two remaining acquisitions for comparison - same budget, other strategy.
    for name in ("PI", "UCB"):
        summarize(name, optimize(name, verbose=False))

    print(f"\nFigure saved to {s.OUT / 'bo-from-scratch.svg'}")


if __name__ == "__main__":
    main()
