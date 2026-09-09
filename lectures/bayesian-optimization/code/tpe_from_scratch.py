"""The tree-structured Parzen estimator (TPE) from scratch - the other model.

The lecture builds Bayesian optimization on a Gaussian process, and a
Gaussian process needs a **distance** between two settings: the kernel
compares ``x`` with ``x'`` through ``||x - x'||``. As soon as one of the
searched variables is a **category** - the route of administration, the
optimizer of a network, a staining protocol - there is no distance, and the
Gaussian process has nothing to compute. TPE (Bergstra, Bardenet, Bengio and
Kegl, 2011) is the model that libraries such as Optuna and Hyperopt run
instead. It is written out here in a few functions so that it can be seen
there is nothing more to it.

**The toy problem** is the dosing schedule from the lecture plus one
categorical variable the Gaussian process could not take:

* the dose per cycle ``d`` in [20, 100] mg/m2 and the interval between cycles
  ``T`` in [10, 35] days - the very landscape of the rest of the lecture;
* the **route of administration**: oral, IV bolus or IV infusion. The toy
  simulator adds a fixed bonus to the score for each route
  (``ROUTE_EFFECT``) - a caricature of "an infusion spreads the peak
  concentration out and is tolerated better". It is not pharmacology, it
  only gives the categorical variable something to decide.

**The method** in four steps, repeated in the usual loop:

1. Split the trials so far into the best quarter (``GAMMA = 0.25``), the
   **good** ones, and the rest, the **bad** ones.
2. Over each group put a density of *settings*, one variable at a time: for
   a number a Gaussian bump on every trial (a Parzen window, that is a kernel
   density estimate), for a category a histogram of the trials. Both get
   **one imaginary trial spread over the whole range** on top, so that no
   density is ever zero and an unexplored region keeps a chance. ``l`` is
   the density of the good trials, ``g`` of the bad ones.
3. Draw candidates from ``l`` and keep the one with the largest ratio
   ``l(x) / g(x)``. Bergstra et al. show that under this model the expected
   improvement is a monotone function of that ratio, so this is expected
   improvement, only computed the other way round.
4. Measure it, add it to the trials, go to 1.

Each variable gets its own pair of densities and the ratios are multiplied:
classic TPE treats the variables as **independent**. That is its weakness -
the trade-off between dose and interval is invisible to it - and also the
reason it does not care what type a variable is: a density over a category
is a histogram, a density over a branch of a conditional search space is
computed only from the trials where that branch existed - hence
*tree-structured*.

Three simplifications against Bergstra et al. (2011), so that the code stays
readable: the bandwidth of a bump is **fixed** (a tenth of the variable's
range; the paper scales it by the distance to the neighbouring trials), the
prior is the **imaginary uniform trial** of weight one in every density
(the paper adds a wide Gaussian), and a category gets the **same recipe**
(counts plus the imaginary trial spread evenly over the categories).

TPE has no ``mu``, so it cannot "recommend the maximum of the model"; like
the libraries, the script recommends the **best measured trial**.

The script prints the **reference run** (seed 7, 4 random starts + 12 TPE
steps; it ends on IV infusion close to the peak), then the **cautionary
run** with seed 4, the seed of the rest of the lecture: there both random
starts that happened to use IV infusion landed on awful doses, the best
quarter was all IV bolus, and TPE - which draws its candidates from the
density of the good trials - never gave the infusion a second chance. A
category is judged only by the trials that happened to use it. Finally a
comparison with random search at the same budget over 40 seeds: how far from
the optimum the recommendation lands and how often the route is right. The
figures of the TPE block (`figures_block5.py`) import everything from here,
so the slides and this script cannot drift apart.

Run it with::

    uv run python lectures/bayesian-optimization/code/tpe_from_scratch.py
"""

import numpy as np

import common as s

# --- the toy problem: dose x interval (continuous) x route (categorical) ----

# The search space. A number has bounds, a category has its values.
SPACE = {
    "dose": ("float", s.BOUNDS[0]),                     # [20, 100] mg/m2
    "interval": ("float", s.BOUNDS[1]),                 # [10, 35] days
    "route": ("cat", ("oral", "IV bolus", "IV infusion")),
}
ROUTE_EFFECT = {"oral": -8.0, "IV bolus": 0.0, "IV infusion": 6.0}   # toy bonus [points]
BEST_ROUTE = max(ROUTE_EFFECT, key=ROUTE_EFFECT.get)
OPTIMUM = {"dose": s.DOSE_OPT, "interval": s.INTERVAL_OPT, "route": BEST_ROUTE}
SCORE_OPT_MIXED = s.SCORE_OPT + ROUTE_EFFECT[BEST_ROUTE]              # 48.13

# --- TPE constants -----------------------------------------------------------

GAMMA = 0.25            # share of trials called "good" (the best quarter)
BANDWIDTH = 0.10        # width of one bump as a fraction of the variable's range
N_CANDIDATES = 24       # candidates drawn from l per step
N_START = 4             # random trials before TPE takes over
N_STEPS = 12            # TPE steps in the reference run
CAUTIONARY_SEED = s.SEED   # 4: the run that locks into the wrong route
SEED = 7                # reference run; seed 4 (the rest of the lecture) is the cautionary run, see below


def true_score(x):
    """The toy simulator without noise: the landscape plus the route bonus."""
    return float(s.score(x["dose"], x["interval"]) + ROUTE_EFFECT[x["route"]])


def measure(x, rng):
    """One **expensive** evaluation: the true score plus the response of one patient."""
    return true_score(x) + float(rng.normal(0.0, s.SIGMA_NOISE))


# --- the two densities -------------------------------------------------------

def split_good_bad(values, gamma=GAMMA):
    """Indices of the good (best quarter) and the bad trials. Larger score is better."""
    values = np.asarray(values, dtype=float)
    n_good = max(1, int(np.ceil(gamma * len(values))))
    order = np.argsort(-values)           # best first
    return order[:n_good], order[n_good:]


def float_density(observed, grid, bounds, bandwidth=BANDWIDTH):
    """Parzen estimate over a number: a Gaussian bump on every observed value
    plus one imaginary trial spread uniformly over the whole range."""
    observed = np.asarray(observed, dtype=float)
    grid = np.asarray(grid, dtype=float)
    width = bounds[1] - bounds[0]
    h = bandwidth * width
    bumps = np.exp(-0.5 * ((grid[:, None] - observed[None, :]) / h) ** 2)
    bumps /= h * np.sqrt(2.0 * np.pi)
    return (bumps.sum(axis=1) + 1.0 / width) / (len(observed) + 1)


def cat_density(observed, categories):
    """Histogram over a category plus one imaginary trial spread evenly over
    the categories - the same recipe as for a number, only without a distance."""
    counts = np.array([sum(1 for v in observed if v == c) for c in categories],
                      dtype=float)
    return (counts + 1.0 / len(categories)) / (len(observed) + 1)


def density(name, observed, query, space=SPACE, bandwidth=BANDWIDTH):
    """Density of one variable evaluated at `query` (numbers or category names)."""
    kind, spec = space[name]
    if kind == "float":
        return float_density(observed, query, spec, bandwidth)
    table = dict(zip(spec, cat_density(observed, spec)))
    return np.array([table[c] for c in query])


# --- one proposal ------------------------------------------------------------

def sample_from_l(name, observed_good, rng, n=N_CANDIDATES, space=SPACE,
                  bandwidth=BANDWIDTH):
    """Draw `n` candidate values of one variable from the density of the good trials."""
    kind, spec = space[name]
    if kind == "cat":
        weights = cat_density(observed_good, spec)
        return rng.choice(spec, size=n, p=weights / weights.sum())
    observed_good = np.asarray(observed_good, dtype=float)
    k = len(observed_good)
    h = bandwidth * (spec[1] - spec[0])
    which = rng.integers(0, k + 1, size=n)       # k + 1 = the imaginary uniform trial
    around_good = observed_good[np.minimum(which, k - 1)] + rng.normal(0.0, h, size=n)
    uniform = rng.uniform(spec[0], spec[1], size=n)
    return np.clip(np.where(which < k, around_good, uniform), spec[0], spec[1])


def propose(trials, values, rng, gamma=GAMMA, bandwidth=BANDWIDTH,
            n_candidates=N_CANDIDATES, space=SPACE):
    """The next trial: the candidate drawn from l with the largest l(x) / g(x).

    Every variable is handled on its own and the ratios are multiplied -
    classic TPE treats the variables as independent.
    """
    good, bad = split_good_bad(values, gamma)
    candidates, ratio = {}, np.ones(n_candidates)
    for name in space:
        column = [t[name] for t in trials]
        observed_good = [column[i] for i in good]
        observed_bad = [column[i] for i in bad]
        candidates[name] = sample_from_l(name, observed_good, rng, n_candidates,
                                         space, bandwidth)
        ratio *= (density(name, observed_good, candidates[name], space, bandwidth)
                  / density(name, observed_bad, candidates[name], space, bandwidth))
    best = int(np.argmax(ratio))
    chosen = {name: candidates[name][best] for name in space}
    return {name: (float(v) if space[name][0] == "float" else str(v))
            for name, v in chosen.items()}


def snapshot(trials, values, gamma=GAMMA, bandwidth=BANDWIDTH, space=SPACE, n_grid=321):
    """Everything TPE knows after the trials so far - for the figures: the
    split, and for every variable the grid (or categories), l and g on it."""
    good, bad = split_good_bad(values, gamma)
    state = {"good": good, "bad": bad,
             "threshold": float(np.min(np.asarray(values)[good]))}
    for name, (kind, spec) in space.items():
        column = [t[name] for t in trials]
        grid = np.linspace(spec[0], spec[1], n_grid) if kind == "float" else list(spec)
        state[name] = {
            "grid": grid,
            "l": density(name, [column[i] for i in good], grid, space, bandwidth),
            "g": density(name, [column[i] for i in bad], grid, space, bandwidth),
        }
    return state


# --- the loop ----------------------------------------------------------------

def random_trial(rng, space=SPACE):
    """One setting drawn uniformly from the search space."""
    x = {}
    for name, (kind, spec) in space.items():
        x[name] = float(rng.uniform(*spec)) if kind == "float" else str(rng.choice(spec))
    return x


def tpe_optimization(n_steps=N_STEPS, n_start=N_START, seed=SEED, gamma=GAMMA,
                     bandwidth=BANDWIDTH):
    """Run TPE over the toy problem. Returns the history of trials and, for
    every step, the state the proposal was computed from (for the figures)."""
    rng = np.random.default_rng(seed)
    trials = [random_trial(rng) for _ in range(n_start)]
    values = [measure(x, rng) for x in trials]
    history = {"trials": [list(trials)], "values": [list(values)],
               "state": [], "proposed": []}
    for _ in range(n_steps):
        history["state"].append(snapshot(trials, values, gamma, bandwidth))
        x = propose(trials, values, rng, gamma, bandwidth)
        history["proposed"].append(x)
        trials.append(x)
        values.append(measure(x, rng))
        history["trials"].append(list(trials))
        history["values"].append(list(values))
    # TPE has no model mean to maximize: recommend the best measured trial.
    best = int(np.argmax(values))
    history["recommendation"] = (trials[best], values[best])
    return history


def random_search(n_total=N_START + N_STEPS, seed=SEED):
    """The baseline with the same budget: every trial drawn at random."""
    rng = np.random.default_rng(seed)
    trials = [random_trial(rng) for _ in range(n_total)]
    values = [measure(x, rng) for x in trials]
    best = int(np.argmax(values))
    return trials[best], values[best]


def compare(n_runs=40, n_total=N_START + N_STEPS):
    """Simple regret of the recommendation, TPE against random search, over
    `n_runs` seeds at the same budget. Returns a dictionary of summaries."""
    regret = {"TPE": [], "random": []}
    right_route = {"TPE": 0, "random": 0}
    for seed in range(n_runs):
        x, _ = tpe_optimization(n_steps=n_total - N_START, seed=100 + seed)["recommendation"]
        regret["TPE"].append(SCORE_OPT_MIXED - true_score(x))
        right_route["TPE"] += x["route"] == BEST_ROUTE
        x, _ = random_search(n_total, seed=100 + seed)
        regret["random"].append(SCORE_OPT_MIXED - true_score(x))
        right_route["random"] += x["route"] == BEST_ROUTE
    return {name: {"median_regret": float(np.median(regret[name])),
                   "within_2": float(np.mean(np.asarray(regret[name]) <= 2.0)),
                   "right_route": right_route[name] / n_runs}
            for name in regret}


def main() -> None:
    print("TPE from scratch - dose x interval (continuous) x route (categorical)")
    print(f"  optimum: {BEST_ROUTE}, d = {s.DOSE_OPT:.0f} mg/m2, T = {s.INTERVAL_OPT:.0f} days, "
          f"true score {SCORE_OPT_MIXED:.2f}")
    print(f"  gamma = {GAMMA}, bandwidth = {BANDWIDTH:.0%} of the range, "
          f"{N_CANDIDATES} candidates per step, seed {SEED}\n")

    run = tpe_optimization()
    final_good = set(int(i) for i in run["state"][-1]["good"]) if run["state"] else set()
    print(f"  {'trial':>5}  {'dose':>6}  {'T':>5}  {'route':<12} {'measured':>9}  {'true':>6}  good?")
    for i, (x, y) in enumerate(zip(run["trials"][-1], run["values"][-1]), start=1):
        tag = "start" if i <= N_START else "TPE"
        print(f"  {i:>5}  {x['dose']:>6.1f}  {x['interval']:>5.1f}  {x['route']:<12} "
              f"{y:>9.2f}  {true_score(x):>6.2f}  {'*' if i - 1 in final_good else ' '}"
              f"   ({tag})")
    x, y = run["recommendation"]
    print(f"\n  recommendation = best measured trial: {x['route']}, "
          f"d = {x['dose']:.1f} mg/m2, T = {x['interval']:.1f} days, "
          f"measured {y:.2f}, true {true_score(x):.2f} "
          f"(regret {SCORE_OPT_MIXED - true_score(x):.2f})")

    caution = tpe_optimization(seed=CAUTIONARY_SEED)
    xc, yc = caution["recommendation"]
    good_routes = sorted({caution["trials"][-1][i]["route"] for i in caution["state"][-1]["good"]})
    print(f"\n  cautionary run (seed {CAUTIONARY_SEED}): recommendation {xc['route']}, "
          f"d = {xc['dose']:.1f}, T = {xc['interval']:.1f}, true {true_score(xc):.2f} "
          f"(regret {SCORE_OPT_MIXED - true_score(xc):.2f}); the good quarter used "
          f"only {', '.join(good_routes)}")

    n_runs = 40
    summary = compare(n_runs)
    print(f"\n  {n_runs} runs, {N_START + N_STEPS} evaluations each, "
          "simple regret of the recommendation:")
    for name, row in summary.items():
        print(f"    {name:<7} median regret {row['median_regret']:.2f} points, "
              f"within 2 points in {100 * row['within_2']:.0f} % of runs, "
              f"right route in {100 * row['right_route']:.0f} %")

    # Guards: the reference run must land on the right route near the peak,
    # and TPE must beat random search - otherwise the slides lie.
    assert x["route"] == BEST_ROUTE, x
    assert abs(x["dose"] - s.DOSE_OPT) < 12.0 and abs(x["interval"] - s.INTERVAL_OPT) < 5.0, x
    assert summary["TPE"]["median_regret"] < summary["random"]["median_regret"]
    assert summary["TPE"]["within_2"] > summary["random"]["within_2"]
    assert summary["TPE"]["right_route"] > summary["random"]["right_route"]
    # ... and the cautionary run must really be the lock-in the notes describe:
    # both infusion starts far from the optimal dose, the good quarter a single
    # wrong route, and the infusion never proposed again.
    starts, later = caution["trials"][-1][:N_START], caution["trials"][-1][N_START:]
    infusion_starts = [t for t in starts if t["route"] == BEST_ROUTE]
    assert len(infusion_starts) == 2 and all(abs(t["dose"] - s.DOSE_OPT) > 30 for t in infusion_starts)
    assert not any(t["route"] == BEST_ROUTE for t in later)
    assert xc["route"] != BEST_ROUTE and good_routes == [xc["route"]], (xc, good_routes)


if __name__ == "__main__":
    main()
