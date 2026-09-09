"""Shared setup, palette and the binding numbers of the lecture's running example.

Main running example: **finding a chemotherapy dosing regimen**. Two numbers
are searched for — the dose per cycle `d` (mg/m2) and the interval between
cycles `T` (days) — and one evaluation means running the patient simulator,
that is, an expensive, noisy black box without a derivative.

The simulator is a **toy, not a pharmacological model**; it only shows the
shape of the trade-off, not real treatment. The score is made up of five
mechanisms:

    S(d, T) = effect(I) - acute(d) - cumulative(I) - burden(T) - regrowth(T)

where `I = d/T` is the dose intensity:

- `effect`     — tumour shrinkage grows with dose intensity `I` and saturates;
- `acute`      — acute toxicity of a single dose, with a threshold around `D0` mg/m2;
- `cumulative` — toxicity from insufficient recovery, grows quadratically with `I`;
- `burden`     — burden on the patient and the clinic, every visit costs something;
- `regrowth`   — tumour repopulation when the pause between cycles is too long.

The last two terms squeeze the interval from both sides, so the optimum is
really a peak, not a ridge — otherwise random search would find it just as
well and the lecture's main figure would not make sense.

The constants `K`, `D0`, `W`, `ALPHA`, `RHO` and `T_REF` are chosen by hand;
`BETA` and `GAMMA` are **derived** so that the gradient of the score is
exactly zero at

    (d*, T*) = (60 mg/m2, 21 days),

that is, at a round number that also happens to match a common three-week
cycle. The optimum is `S* = 42.13` points. That it comes out at a round
regimen is a consequence of the choice of constants — the slides say so out
loud.
"""

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import erf

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams["svg.hashsalt"] = "aui-bayesian-optimization"
SVG_METADATA = {"Date": None}

# The palette is shared across the whole course.
BLUE = "#12355b"
TEAL = "#007f86"
RED = "#c73e1d"
ORANGE = "#e9a23b"
LIGHT_BLUE = "#d9eef7"
GREY = "#eef2f4"

# --- Patient simulator: binding constants ----------------------------------

K = 0.55        # how fast the effect saturates with dose intensity
D0 = 85.0       # acute-toxicity threshold [mg/m2]
W = 6.0         # width of the threshold [mg/m2]
ALPHA = 40.0    # weight of acute toxicity
RHO = 0.05      # weight of tumour repopulation over a long pause
T_REF = 14.0    # interval from which regrowth starts to show [days]

DOSE_OPT = 60.0       # d* [mg/m2]
INTERVAL_OPT = 21.0   # T* [days]

BOUNDS = ((20.0, 100.0), (10.0, 35.0))   # (d_min, d_max), (T_min, T_max)
SIGMA_NOISE = 2.0                        # standard deviation of the patient response
SEED = 4


def _derive_constants():
    """Return (BETA, GAMMA) such that the gradient of the score at the optimum is zero.

    Stationarity in two variables, after substituting `I = d/T` and writing
    `P = 100 K exp(-K I) - 2 BETA I`, gives:

        P / T  = ALPHA * sigma'(d)                       (derivative w.r.t. dose)
        P * d  = 21 GAMMA - 2 RHO (T - T_REF) T^2         (derivative w.r.t. interval)

    The first equation gives `P`, and with it `BETA`; the second then gives `GAMMA`.
    """
    d, t = DOSE_OPT, INTERVAL_OPT
    intensity = d / t
    effect_slope = 100.0 * K * np.exp(-K * intensity)
    sig = 1.0 / (1.0 + np.exp(-(d - D0) / W))
    sig_derivative = sig * (1.0 - sig) / W

    p = t * ALPHA * sig_derivative
    beta = (effect_slope - p) / (2.0 * intensity)
    gamma = (p * d + 2.0 * RHO * (t - T_REF) * t**2) / 21.0
    return beta, gamma


BETA, GAMMA = _derive_constants()   # 1.6312 and 20.7136


def score(dose, interval):
    """Score of a regimen: tumour shrinkage minus toxicity minus burden. No noise."""
    dose = np.asarray(dose, dtype=float)
    interval = np.asarray(interval, dtype=float)
    intensity = dose / interval
    effect = 100.0 * (1.0 - np.exp(-K * intensity))
    acute = ALPHA / (1.0 + np.exp(-(dose - D0) / W))
    cumulative = BETA * intensity**2
    burden = GAMMA * (21.0 / interval)
    regrowth = RHO * np.maximum(0.0, interval - T_REF) ** 2
    return effect - acute - cumulative - burden - regrowth


SCORE_OPT = float(score(DOSE_OPT, INTERVAL_OPT))   # 42.13


def measure(dose, interval, rng):
    """One **expensive** evaluation: the score plus the response of one patient."""
    return score(dose, interval) + rng.normal(0.0, SIGMA_NOISE,
                                              size=np.shape(dose))


# --- Gaussian process: fifteen lines, no library ---------------------------
#
# Inputs are rescaled to the unit cube before modelling, because mg/m2 and
# days are not comparable units and a single length scale would have to
# serve both.
#
# The kernel's length scale and amplitude are not guessed: they are
# **rounded** values from the maximum of the marginal likelihood on 40
# random measurements of the patient simulator (the maximum lies at
# l = 0.385 and A = 44.2). `kernel_estimation.py` recomputes both the
# estimate and the fact that rounding costs nothing.
# On the slides this is called "the kernel is itself a hyperparameter".

KERNEL_LENGTH = 0.38   # in units of the rescaled cube
AMPLITUDE = 43.0       # standard deviation of the prior [score points]
KAPPA_UCB = 2.0        # exploration/exploitation trade-off of the UCB acquisition


def to_unit_cube(points):
    """Rescale points from physical units to the cube [0, 1]^2."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    lower = np.array([BOUNDS[0][0], BOUNDS[1][0]])
    upper = np.array([BOUNDS[0][1], BOUNDS[1][1]])
    return (points - lower) / (upper - lower)


def rbf_kernel(a, b, length=KERNEL_LENGTH, amplitude=AMPLITUDE):
    """Gaussian (RBF) kernel: k(x, x') = A^2 exp(-||x - x'||^2 / (2 l^2))."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return amplitude**2 * np.exp(-0.5 * d2 / length**2)


def gp_posterior(x_data, y_data, x_query, length=KERNEL_LENGTH, amplitude=AMPLITUDE,
                 noise=SIGMA_NOISE, mean=None):
    """Posterior mean and standard deviation of the Gaussian process.

    Returns a pair of arrays `(mu, sigma)` at the points `x_query`. Inputs are
    expected to already be rescaled to the unit cube.
    """
    x_data = np.atleast_2d(np.asarray(x_data, dtype=float))
    y_data = np.asarray(y_data, dtype=float)
    x_query = np.atleast_2d(np.asarray(x_query, dtype=float))
    mean = float(y_data.mean()) if mean is None else float(mean)

    kk = rbf_kernel(x_data, x_data, length, amplitude)
    kk = kk + (noise**2 + 1e-8) * np.eye(len(x_data))
    ks = rbf_kernel(x_data, x_query, length, amplitude)
    solution = np.linalg.solve(kk, np.column_stack([y_data - mean, ks]))
    alpha, v = solution[:, 0], solution[:, 1:]

    mu = mean + ks.T @ alpha
    variance = amplitude**2 - np.einsum("ij,ij->j", ks, v)
    return mu, np.sqrt(np.clip(variance, 1e-12, None))


# --- Acquisition functions --------------------------------------------------


def _phi(z):
    """CDF of the standard normal distribution."""
    return 0.5 * (1.0 + erf(np.asarray(z) / np.sqrt(2.0)))


def _phi_density(z):
    return np.exp(-0.5 * np.asarray(z) ** 2) / np.sqrt(2.0 * np.pi)


def acquisition_pi(mu, sigma, best, xi=0.01):
    """Probability of improvement."""
    z = (mu - best - xi) / sigma
    return _phi(z)


def acquisition_ei(mu, sigma, best, xi=0.01):
    """Expected improvement."""
    diff = mu - best - xi
    z = diff / sigma
    return diff * _phi(z) + sigma * _phi_density(z)


def acquisition_ucb(mu, sigma, best=None, kappa=KAPPA_UCB):
    """Upper confidence bound (optimism in the face of uncertainty)."""
    return mu + kappa * sigma


ACQUISITIONS = {"PI": acquisition_pi, "EI": acquisition_ei, "UCB": acquisition_ucb}


# --- The whole Bayesian optimization loop ----------------------------------


def candidate_grid(n=61):
    """Regular grid of candidate points in physical units."""
    d = np.linspace(*BOUNDS[0], n)
    t = np.linspace(*BOUNDS[1], n)
    dd, tt = np.meshgrid(d, t, indexing="ij")
    return d, t, np.column_stack([dd.ravel(), tt.ravel()])


def bayesian_optimization(n_steps=8, n_start=4, acquisition="EI",
                          kappa=KAPPA_UCB, seed=SEED, length=KERNEL_LENGTH,
                          amplitude=AMPLITUDE):
    """Run Bayesian optimization over the patient simulator.

    Returns a dictionary with the history: the points visited, the measured
    scores, and at every step also the model (mu, sigma) and the acquisition
    on the grid, so an animation can be built from it.
    """
    rng = np.random.default_rng(seed)
    _, _, candidates = candidate_grid()

    # Random start (Latin-hypercube-like: evenly spread in both coordinates).
    points = np.column_stack([
        rng.permutation(np.linspace(*BOUNDS[0], n_start + 2)[1:-1]),
        rng.permutation(np.linspace(*BOUNDS[1], n_start + 2)[1:-1]),
    ])
    values = measure(points[:, 0], points[:, 1], rng)

    history = {"points": [points.copy()], "values": [values.copy()],
               "mu": [], "sigma": [], "acquisition": [], "selected": []}
    fun = ACQUISITIONS[acquisition]

    for _ in range(n_steps):
        mu, sigma = gp_posterior(to_unit_cube(points), values,
                                 to_unit_cube(candidates), length, amplitude)
        if acquisition == "UCB":
            a = fun(mu, sigma, kappa=kappa)
        else:
            a = fun(mu, sigma, values.max())
        new_point = candidates[int(np.argmax(a))]

        history["mu"].append(mu)
        history["sigma"].append(sigma)
        history["acquisition"].append(a)
        history["selected"].append(new_point.copy())

        points = np.vstack([points, new_point])
        values = np.append(values, measure(new_point[0], new_point[1], rng))
        history["points"].append(points.copy())
        history["values"].append(values.copy())

    history["recommendation"] = recommendation_from_model(points, values, length,
                                                           amplitude)
    return history


def recommendation_from_model(points, values, length=KERNEL_LENGTH,
                              amplitude=AMPLITUDE):
    """Recommended regimen = maximum of the model's mean, not the last measurement."""
    _, _, candidates = candidate_grid()
    mu, _ = gp_posterior(to_unit_cube(points), values,
                         to_unit_cube(candidates), length, amplitude)
    return candidates[int(np.argmax(mu))]


# --- Saving figures ----------------------------------------------------------


def save_figure(fig, name: str) -> None:
    """Save the figure as SVG; with the AUI_PREVIEWS variable also as PNG, for checking."""
    fig.savefig(OUT / f"{name}.svg", metadata=SVG_METADATA, bbox_inches="tight")
    previews = os.environ.get("AUI_PREVIEWS")
    if previews:
        directory = Path(previews)
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_gif(frames, durations, name: str, dpi: int = 120) -> None:
    """Save a sequence of figures as a looping GIF.

    `frames` are the finished figures in playback order, `durations` their
    display times in milliseconds. **The first frame also serves as the
    static fallback** (in a PDF or when printed only that one is shown), so
    it should show the final state.
    """
    images = []
    for fig in frames:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("RGB")
                      .quantize(colors=96, method=Image.MEDIANCUT))
    images[0].save(OUT / f"{name}.gif", save_all=True, append_images=images[1:],
                   duration=list(durations), loop=0, optimize=True, disposal=2)


def landscape(n=241):
    """Grid of scores for the contour figures: returns (d, T, S)."""
    d = np.linspace(*BOUNDS[0], n)
    t = np.linspace(*BOUNDS[1], n)
    dd, tt = np.meshgrid(d, t, indexing="ij")
    return d, t, score(dd, tt)


# Shared contour levels — the dynamic range of the score is large (the
# high-dose, short-interval corner goes deep into negative territory), so the
# levels are chosen by hand and kept the same across all figures.
LEVELS = np.array([-40, -20, -10, 0, 10, 20, 28, 34, 38, 41], dtype=float)
