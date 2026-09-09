"""The patient simulator as an expensive black box - and a check-off printout.

The script does two things at once:

1. **Shows the black box** that the whole lecture optimizes: it gets a dose
   and an interval, returns one noisy score and nothing else - no
   derivative, no formula.
2. **Fixes the numbers** that appear on the slides. Every value that shows up
   in the lecture as a concrete number is printed by this script; the slides
   copy it, they do not recompute it.

Run it from the repository root::

    uv run python lectures/bayesian-optimization/code/patient_simulator.py
"""

import numpy as np

import common as s


def heading(title):
    print(f"\n{title}\n" + "-" * len(title))


def main():
    np.set_printoptions(precision=3, suppress=True)

    heading("Simulator: constants")
    print(f"K = {s.K}, D0 = {s.D0} mg/m2, W = {s.W}, ALPHA = {s.ALPHA}")
    print(f"BETA = {s.BETA:.6f}   (derived)")
    print(f"GAMMA = {s.GAMMA:.6f}   (derived)")
    print(f"bounds: dose {s.BOUNDS[0]} mg/m2, interval {s.BOUNDS[1]} days")
    print(f"patient response noise: sigma = {s.SIGMA_NOISE}")

    heading("Optimum")
    d, t = s.DOSE_OPT, s.INTERVAL_OPT
    step = 1e-5
    grad = np.array([
        (s.score(d + step, t) - s.score(d - step, t)) / (2 * step),
        (s.score(d, t + step) - s.score(d, t - step)) / (2 * step),
    ])
    print(f"(d*, T*) = ({d:.0f} mg/m2, {t:.0f} days),  S* = {s.SCORE_OPT:.4f}")
    print(f"gradient at the optimum = {grad}  (should be zero)")
    assert np.allclose(grad, 0.0, atol=1e-6), "optimum is not a stationary point"

    dd, tt, S = s.landscape(801)
    i = np.unravel_index(np.argmax(S), S.shape)
    print(f"maximum on the fine grid: d = {dd[i[0]]:.3f}, T = {tt[i[1]]:.3f}, "
          f"S = {S[i]:.4f}")
    print(f"score range over the region: {S.min():.1f} to {S.max():.1f}")

    heading("Score at selected regimens (noise-free)")
    print(f"{'dose':>7} {'interval':>9} {'intensity':>10} {'score':>8}")
    for dose, interval in [(60, 21), (60, 14), (60, 28), (40, 21), (80, 21),
                           (100, 21), (30, 10), (100, 10), (20, 35)]:
        print(f"{dose:7.0f} {interval:9.0f} {dose / interval:10.2f} "
              f"{float(s.score(dose, interval)):8.2f}")

    heading("Score breakdown at the optimum")
    intensity = d / t
    effect = 100.0 * (1.0 - np.exp(-s.K * intensity))
    acute = s.ALPHA / (1.0 + np.exp(-(d - s.D0) / s.W))
    cumulative = s.BETA * intensity**2
    burden = s.GAMMA * (21.0 / t)
    regrowth = s.RHO * max(0.0, t - s.T_REF) ** 2
    print(f"effect      +{effect:6.2f}")
    print(f"acute tox.  -{acute:6.2f}")
    print(f"cumul. tox. -{cumulative:6.2f}")
    print(f"burden      -{burden:6.2f}")
    print(f"regrowth    -{regrowth:6.2f}")
    print(f"total        {effect - acute - cumulative - burden - regrowth:6.2f}")

    heading("Bayesian optimization: 4 starting points + 8 steps, EI acquisition")
    history = s.bayesian_optimization()
    points, values = history["points"][-1], history["values"][-1]
    for i, (point, value) in enumerate(zip(points, values)):
        label = "start" if i < 4 else f"step {i - 3}"
        print(f"{label:>7}: d = {point[0]:6.1f}, T = {point[1]:5.1f}  ->  "
              f"measured {value:7.2f}   (noise-free {float(s.score(*point)):7.2f})")
    best = int(np.argmax(values))
    print(f"\nbest measured point:  d = {points[best][0]:.1f}, "
          f"T = {points[best][1]:.1f}, measured {values[best]:.2f}")
    rec = history["recommendation"]
    print(f"recommendation from the model:   d = {rec[0]:.1f}, T = {rec[1]:.1f}, "
          f"true score {float(s.score(*rec)):.2f}")
    print(f"(the optimum is {s.SCORE_OPT:.2f}, loss "
          f"{s.SCORE_OPT - float(s.score(*rec)):.2f} points after 12 evaluations)")

    heading("Three candidates for a manual acquisition calculation (state after 6 evaluations)")
    points6 = history["points"][2]
    values6 = history["values"][2]
    mu, sigma = s.gp_posterior(s.to_unit_cube(points6), values6,
                               s.to_unit_cube(CANDIDATES))
    best_value = float(values6.max())
    print(f"best measured score so far f+ = {best_value:.2f}")
    print(f"{'candidate':>18} {'mu':>7} {'sigma':>7} {'PI':>7} {'EI':>7} "
          f"{'UCB':>7} {'true':>9}")
    for cand, m, sg in zip(CANDIDATES, mu, sigma):
        pi = float(s.acquisition_pi(m, sg, best_value, xi=0.0))
        ei = float(s.acquisition_ei(m, sg, best_value, xi=0.0))
        ucb = float(s.acquisition_ucb(m, sg))
        print(f"  d={cand[0]:5.0f} T={cand[1]:4.0f} {m:7.2f} {sg:7.2f} "
              f"{pi:7.3f} {ei:7.2f} {ucb:7.2f} {float(s.score(*cand)):9.2f}")
    print("All three acquisitions put candidate A first, but they disagree on")
    print("second place: PI and EI prefer the more certain C, UCB bets on the")
    print("unexplored B. That is exactly what distinguishes the three.")

    heading("Grid x random x Bayesian optimization (30 evaluations, 40 runs)")
    results = strategy_comparison()
    print(f"{'n':>4} " + " ".join(f"{j:>17}" for j in results))
    for n in (5, 10, 15, 20, 30):
        print(f"{n:4d} " + " ".join(f"{v[n - 1]:17.2f}" for v in results.values()))
    print(f"\n(the optimum is {s.SCORE_OPT:.2f})")

    print(f"\nhow many evaluations are needed to reach a score of {TARGET:.0f} points:")
    for name, value in evaluations_to_target().items():
        median, success = value
        print(f"{name:>17}: median {median:5.1f}, successful runs {success:3.0f} %")


# Three candidates for a manual acquisition calculation - chosen so that the
# acquisition functions disagree: A has a high mean and low uncertainty, B is
# in an unexplored corner, C lies between them. The lecture ends up without a
# slide of manual acquisition arithmetic (see `sources.md`,
# archived question 3), so these numbers stay here as a backup for the lab or the
# whiteboard.
CANDIDATES = np.array([[55.0, 24.0],    # A - high mu, medium uncertainty
                       [25.0, 20.0],    # B - low mu, high uncertainty
                       [45.0, 18.0]])   # C - medium mu, low uncertainty

TARGET = 41.0        # target score for comparing strategies
N_COMPARISON = 30
N_REPEATS = 40


def _best_so_far(points, values):
    """True (noise-free) score of the best point measured so far."""
    return np.maximum.accumulate(
        [float(s.score(*points[np.argmax(values[:i + 1])]))
         for i in range(len(values))])


def _runs():
    """Return arrays (repeats x count) for grid, random, EI and UCB."""
    random_search = np.zeros((N_REPEATS, N_COMPARISON))
    ei = np.zeros_like(random_search)
    ucb = np.zeros_like(random_search)
    for b in range(N_REPEATS):
        rng = np.random.default_rng(2000 + b)
        points = np.column_stack([rng.uniform(*s.BOUNDS[0], N_COMPARISON),
                                  rng.uniform(*s.BOUNDS[1], N_COMPARISON)])
        random_search[b] = _best_so_far(points, s.measure(points[:, 0], points[:, 1], rng))
        for array, acquisition in ((ei, "EI"), (ucb, "UCB")):
            history = s.bayesian_optimization(n_steps=N_COMPARISON - 4,
                                              seed=2000 + b, acquisition=acquisition)
            array[b] = _best_so_far(history["points"][-1], history["values"][-1])

    # The grid always places its points the same way, but the patient
    # response is noisy - so it also gets its forty runs. Without that the
    # fourth panel of `strategy-comparison.gif` would promise an average
    # over forty runs and draw a single one for the grid.
    dm = np.linspace(*s.BOUNDS[0], 7)[1:-1]
    tm = np.linspace(*s.BOUNDS[1], 8)[1:-1]
    grid_points = np.array([(a, b) for a in dm for b in tm])[:N_COMPARISON]
    grid = np.zeros_like(random_search)
    for b in range(N_REPEATS):
        rng = np.random.default_rng(2000 + b)
        grid[b] = _best_so_far(
            grid_points, s.measure(grid_points[:, 0], grid_points[:, 1], rng))
    return grid, random_search, ei, ucb


def strategy_comparison():
    """Best score found as a function of the number of evaluations (averaged over runs)."""
    grid, random_search, ei, ucb = _runs()
    return {"grid 5x6": grid.mean(0),
            "random search": random_search.mean(0),
            "Bayesian (EI)": ei.mean(0),
            "Bayesian (UCB)": ucb.mean(0)}


def evaluations_to_target():
    """How many evaluations are needed to reach score `TARGET`."""
    _, random_search, ei, ucb = _runs()
    results = {}
    for name, array in (("random search", random_search), ("Bayesian (EI)", ei),
                        ("Bayesian (UCB)", ucb)):
        counts = []
        for row in array:
            i = int(np.argmax(row >= TARGET))
            counts.append(i + 1 if row[i] >= TARGET else N_COMPARISON + 1)
        counts = np.array(counts)
        results[name] = (float(np.median(counts)),
                         float((counts <= N_COMPARISON).mean() * 100))
    return results


if __name__ == "__main__":
    main()
