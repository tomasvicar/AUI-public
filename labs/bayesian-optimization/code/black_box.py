"""The patient simulator as an expensive black box - shared by the whole lab.

This is a **copy** of the simulator from the lecture
(`lectures/bayesian-optimization/code/common.py`), and deliberately so: the
notebooks run in Colab, where the rest of the repository does not exist, so the
simulator has to fit inside the notebook. To stop the copy from drifting away
from the original, this script can compare itself against the lecture:

    uv run python labs/bayesian-optimization/code/black_box.py

It prints the reference numbers and - when the lecture is at hand - checks that
the score matches the original to 1e-12 on a dense grid.

What the simulator does: it takes a dose per cycle `d` [mg/m2] and an interval
between cycles `T` [days] and returns **one noisy number**. No derivative, no
formula. The optimum is (60 mg/m2, 21 days) with a score of 42.13 points; that
is the one thing the teacher knows and the students do not.
"""

import numpy as np

# --- Binding constants (copied from the lecture, do not change) ------------

K = 0.55        # how fast the effect saturates with dose intensity
D0 = 85.0       # threshold of acute toxicity [mg/m2]
W = 6.0         # width of the threshold [mg/m2]
ALPHA = 40.0     # weight of acute toxicity
RHO = 0.05       # weight of tumour repopulation during a long pause
T_REF = 14.0    # interval beyond which repopulation shows up [days]

DOSE_OPT = 60.0      # d* [mg/m2]
INTERVAL_OPT = 21.0   # T* [days]

BOUNDS = ((20.0, 100.0), (10.0, 35.0))   # (d_min, d_max), (T_min, T_max)
LOW, HIGH = np.array(BOUNDS).T           # the corners of the region
SIGMA_NOISE = 2.0                        # standard deviation of the patient response


def _derive_constants():
    """BETA and GAMMA such that the gradient of the score at (60, 21) is zero."""
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
    """True (noise-free) score of a regimen - what infinitely many patients would give."""
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
    """One **expensive** evaluation: the score plus the response of one patient.

    In reality this line would be a twenty-minute simulation or a week of
    measurement; here it is a millisecond, but we treat it as expensive - we
    count how many times it was called.
    """
    return score(dose, interval) + rng.normal(0.0, SIGMA_NOISE,
                                               size=np.shape(dose))


def random_points(rng, count):
    """`count` uniformly random points inside the bounds, shape (count, 2)."""
    return LOW + rng.random((count, 2)) * (HIGH - LOW)


# --- Check against the lecture ---------------------------------------------


def _compare_with_lecture():
    """Verify that the copy computes the same thing as the original."""
    import importlib.util
    from pathlib import Path

    lecture_path = (Path(__file__).resolve().parents[3]
             / "lectures" / "bayesian-optimization" / "code" / "common.py")
    if not lecture_path.exists():
        return f"skipped - {lecture_path} is not here (Colab?)"

    spec = importlib.util.spec_from_file_location("_lecture", lecture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dd = np.linspace(*BOUNDS[0], 97)
    tt = np.linspace(*BOUNDS[1], 89)
    grid_d, grid_t = np.meshgrid(dd, tt, indexing="ij")
    ours = score(grid_d, grid_t)
    lecture_val = module.score(grid_d, grid_t)
    deviation = float(np.abs(ours - lecture_val).max())
    assert deviation < 1e-12, f"the copy drifted from the lecture by {deviation}"
    for name_here, name_there in (("K", "K"), ("D0", "D0"), ("W", "W"),
                                  ("ALPHA", "ALPHA"), ("RHO", "RHO"),
                                  ("T_REF", "T_REF"), ("SIGMA_NOISE", "SIGMA_NOISE")):
        assert getattr(module, name_there) == globals()[name_here], \
            f"constant {name_here} differs"
    return f"OK - largest deviation from the lecture {deviation:.1e}"


def main():
    print("Patient simulator - reference numbers")
    print("-" * 38)
    print(f"BETA = {BETA:.4f}, GAMMA = {GAMMA:.4f}")
    print(f"optimum (d*, T*) = ({DOSE_OPT:.0f} mg/m2, {INTERVAL_OPT:.0f} days)")
    print(f"score at the optimum = {SCORE_OPT:.4f} points")

    step = 1e-5
    grad = np.array([
        (score(DOSE_OPT + step, INTERVAL_OPT) - score(DOSE_OPT - step, INTERVAL_OPT)) / (2 * step),
        (score(DOSE_OPT, INTERVAL_OPT + step) - score(DOSE_OPT, INTERVAL_OPT - step)) / (2 * step),
    ])
    print(f"gradient at the optimum = {grad} (must be zero)")
    assert np.allclose(grad, 0.0, atol=1e-6), "the optimum is not a stationary point"

    dd = np.linspace(*BOUNDS[0], 801)
    tt = np.linspace(*BOUNDS[1], 801)
    grid = score(dd[:, None], tt[None, :])
    i = np.unravel_index(np.argmax(grid), grid.shape)
    print(f"maximum on a fine grid: d = {dd[i[0]]:.3f}, T = {tt[i[1]]:.3f}, "
          f"S = {grid[i]:.4f}")
    print(f"range of the score over the region: {grid.min():.1f} to {grid.max():.1f}")

    rng = np.random.default_rng(0)
    sample = measure(np.full(10_000, DOSE_OPT), np.full(10_000, INTERVAL_OPT), rng)
    print(f"noise: 10 000 measurements at the optimum have mean {sample.mean():.3f} "
          f"and sd {sample.std():.3f} (expected {SCORE_OPT:.2f} and {SIGMA_NOISE})")
    # Both moments are checked: the mean would catch a shifted simulator, the
    # standard deviation a simulator that forgot to add noise at all.
    assert abs(sample.mean() - SCORE_OPT) < 0.1, "the noise shifts the mean"
    assert abs(sample.std() - SIGMA_NOISE) < 0.1, "the noise has the wrong width"

    print("\nAgreement with the lecture:", _compare_with_lecture())


if __name__ == "__main__":
    main()
