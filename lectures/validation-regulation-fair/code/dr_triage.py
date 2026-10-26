"""The running example: a diabetic-retinopathy triage software in a screening
programme. Every number of the lecture is computed here.

The model is not trained. Its score is described by the textbook binormal
model: the score of a patient without referable retinopathy is N(0, 1), of a
patient with it N(d, 1), where d is chosen so that the AUC is 0.94. That keeps
the example exact (no sampling noise) and the students can check every number
by hand. The prevalence of referable retinopathy in the screened population
is an assumption of the example, stated on the slide.

Run from the repository root:

    uv run python lectures/validation-regulation-fair/code/dr_triage.py
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score

# --- the screening population and the model ---------------------------------

AUC = 0.94
PREVALENCE = 0.08          # referable DR among screened diabetic patients (assumed)
SCREENED = 1000            # patients per worked example
TARGET_SENSITIVITY = 0.90  # the operating point the start-up chose
D_PRIME = np.sqrt(2) * norm.ppf(AUC)  # equal-variance binormal: AUC = Phi(d / sqrt 2)


def rates_at_score(threshold: float) -> tuple[float, float]:
    """Sensitivity and specificity when everyone with score >= threshold is referred."""
    return float(norm.sf(threshold - D_PRIME)), float(norm.cdf(threshold))


def score_for_risk(risk: float) -> float:
    """The score at which the calibrated risk of referable DR equals `risk`."""
    log_odds = np.log(risk / (1 - risk)) - np.log(PREVALENCE / (1 - PREVALENCE))
    return float((log_odds + D_PRIME**2 / 2) / D_PRIME)


def risk_at_score(score: float) -> float:
    """Calibrated probability of referable DR given the score (Bayes' rule)."""
    sick = PREVALENCE * norm.pdf(score - D_PRIME)
    healthy = (1 - PREVALENCE) * norm.pdf(score)
    return float(sick / (sick + healthy))


def net_benefit_model(threshold_risk: float) -> float:
    """Net benefit of referring everyone whose risk is >= threshold_risk."""
    sens, spec = rates_at_score(score_for_risk(threshold_risk))
    w = threshold_risk / (1 - threshold_risk)
    return sens * PREVALENCE - (1 - spec) * (1 - PREVALENCE) * w


def net_benefit_refer_all(threshold_risk: float) -> float:
    w = threshold_risk / (1 - threshold_risk)
    return PREVALENCE - (1 - PREVALENCE) * w


def counts(sens: float, spec: float, n: int = SCREENED) -> dict[str, int]:
    """Rounded 2x2 table for n screened patients."""
    sick = round(n * PREVALENCE)
    healthy = n - sick
    tp = round(sens * sick)
    tn = round(spec * healthy)
    return {"sick": sick, "healthy": healthy, "tp": tp, "fn": sick - tp, "tn": tn, "fp": healthy - tn}


# the operating point: sensitivity 90 %
OP_SCORE = D_PRIME - norm.ppf(TARGET_SENSITIVITY)
OP_SENS, OP_SPEC = rates_at_score(OP_SCORE)
OP_RISK = risk_at_score(OP_SCORE)
OP_COUNTS = counts(OP_SENS, OP_SPEC)

# the worked net-benefit example: refer if the risk is at least 5 %
# (a missed referable DR is weighed as 19 unnecessary referrals)
NB_THRESHOLD = 0.05
NB_SENS, NB_SPEC = rates_at_score(score_for_risk(NB_THRESHOLD))
NB_COUNTS = counts(NB_SENS, NB_SPEC)
NB_MODEL = net_benefit_model(NB_THRESHOLD)
NB_ALL = net_benefit_refer_all(NB_THRESHOLD)

# decision curve
DC_THRESHOLDS = np.linspace(0.01, 0.40, 79)
DC_MODEL = np.array([net_benefit_model(t) for t in DC_THRESHOLDS])
DC_ALL = np.array([net_benefit_refer_all(t) for t in DC_THRESHOLDS])

# --- how many positives a validation study needs ------------------------------

CI_SENSITIVITY = 0.95
CI_POSITIVES = np.array([20, 40, 60, 100, 200, 300, 400, 600, 800])  # 95 % of each is a whole number


def wilson(successes: int, n: int, z: float = float(norm.ppf(0.975))) -> tuple[float, float]:
    """Wilson score 95 % interval of a proportion."""
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


CI_EXAMPLE = wilson(38, 40)  # "sensitivity 95 % on 40 positives"
CI_CURVE = [wilson(round(CI_SENSITIVITY * n), int(n)) for n in CI_POSITIVES]


def positives_for_half_width(half_width: float, p: float = CI_SENSITIVITY) -> int:
    """Smallest number of positives whose Wilson 95 % interval is at most +-half_width wide."""
    for n in range(10, 5000):
        low, high = wilson(round(p * n), n)
        if (high - low) / 2 <= half_width:
            return n
    raise ValueError("not reached")


POSITIVES_FOR_5PCT = positives_for_half_width(0.05)
SCREENED_FOR_5PCT = int(np.ceil(POSITIVES_FOR_5PCT / PREVALENCE))

# --- two graders on the five DR grades (0 none ... 4 proliferative) -----------

GRADES = ["0 none", "1 mild", "2 moderate", "3 severe", "4 proliferative"]
# rows: grader A, columns: grader B; 200 fundus images. Hand-set so that
# most disagreements are by one grade, as in real grading.
GRADER_TABLE = np.array([
    [88, 9, 1, 0, 0],
    [7, 18, 5, 0, 0],
    [1, 6, 31, 4, 0],
    [0, 0, 4, 11, 2],
    [0, 0, 0, 2, 11],
])


def _labels_from_table(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            a += [i] * int(table[i, j])
            b += [j] * int(table[i, j])
    return np.array(a), np.array(b)


GRADER_A, GRADER_B = _labels_from_table(GRADER_TABLE)
AGREEMENT = float(np.trace(GRADER_TABLE) / GRADER_TABLE.sum())
KAPPA = float(cohen_kappa_score(GRADER_A, GRADER_B))
KAPPA_QUADRATIC = float(cohen_kappa_score(GRADER_A, GRADER_B, weights="quadratic"))
# agreement on the triage decision itself: referable = grade 2 or higher
REFERABLE_A, REFERABLE_B = GRADER_A >= 2, GRADER_B >= 2
REFERABLE_AGREEMENT = float(np.mean(REFERABLE_A == REFERABLE_B))
REFERABLE_KAPPA = float(cohen_kappa_score(REFERABLE_A, REFERABLE_B))

# --- published counts used on the evidence slide --------------------------------

# van Leeuwen et al. 2021 (Eur Radiol 31:3797) and Antonissen et al. 2025 (Eur Radiol):
# share of CE-marked radiology AI products with any peer-reviewed evidence of
# efficacy, and with evidence at level 3 or higher (Fryback & Thornbury).
EVIDENCE = {
    "2020": {"products": 100, "any": 0.36, "level3plus": 0.18},
    "2023": {"products": 173, "any": 0.66, "level3plus": 0.31},
}


def _checks() -> None:
    assert abs(norm.cdf(D_PRIME / np.sqrt(2)) - AUC) < 1e-12
    assert abs(OP_SENS - TARGET_SENSITIVITY) < 1e-12
    # the refer-all line crosses zero exactly at the prevalence
    assert abs(net_benefit_refer_all(PREVALENCE)) < 1e-12
    # the model beats both default strategies over the whole plotted range
    assert np.all(DC_MODEL > np.maximum(DC_ALL, 0))
    # the score/risk mapping is consistent
    assert abs(risk_at_score(score_for_risk(0.2)) - 0.2) < 1e-9
    assert GRADER_TABLE.sum() == 200
    assert KAPPA < KAPPA_QUADRATIC, "near misses should count less with weights"


def report() -> None:
    _checks()
    print(f"binormal d' = {D_PRIME:.4f} for AUC {AUC}")
    print(f"prevalence {PREVALENCE:.0%}, {SCREENED} screened")
    print(f"operating point: sensitivity {OP_SENS:.1%}, specificity {OP_SPEC:.1%}, "
          f"calibrated risk at the threshold {OP_RISK:.1%}")
    c = OP_COUNTS
    print(f"  per {SCREENED}: sick {c['sick']} (TP {c['tp']}, FN {c['fn']}), healthy {c['healthy']} "
          f"(TN {c['tn']}, FP {c['fp']}); referred {c['tp'] + c['fp']}")
    c = NB_COUNTS
    print(f"net benefit at risk threshold {NB_THRESHOLD:.0%} (weight 1:{(1 - NB_THRESHOLD) / NB_THRESHOLD:.0f}): "
          f"sensitivity {NB_SENS:.1%}, specificity {NB_SPEC:.1%}")
    print(f"  per {SCREENED}: TP {c['tp']}, FP {c['fp']}, FN {c['fn']}; referred {c['tp'] + c['fp']}")
    print(f"  NB model {NB_MODEL:.4f} = {NB_MODEL * SCREENED:.1f} net true referrals per {SCREENED}; "
          f"refer all {NB_ALL:.4f} = {NB_ALL * SCREENED:.1f}; refer none 0")
    print(f"Wilson 95 % CI for 38/40: {CI_EXAMPLE[0]:.1%} to {CI_EXAMPLE[1]:.1%}")
    for n, (lo, hi) in zip(CI_POSITIVES, CI_CURVE):
        print(f"  {n:4d} positives: {lo:.1%} to {hi:.1%}")
    print(f"positives for +-5 %: {POSITIVES_FOR_5PCT}, i.e. about {SCREENED_FOR_5PCT} screened at "
          f"{PREVALENCE:.0%} prevalence")
    print(f"graders: raw agreement {AGREEMENT:.1%}, kappa {KAPPA:.3f}, quadratic-weighted kappa "
          f"{KAPPA_QUADRATIC:.3f}; referable yes/no agreement {REFERABLE_AGREEMENT:.1%}, "
          f"kappa {REFERABLE_KAPPA:.3f}")


if __name__ == "__main__":
    report()
