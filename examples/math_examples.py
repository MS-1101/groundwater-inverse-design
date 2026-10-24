"""Executable mathematical checks, separate from groundwater simulation results.

No third-party dependencies. Numbers printed here are teaching examples.
"""
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def gaussian_expected_risk(v, mu=0.0, v0=1.0, threshold=1.0,
                           false_negative=9.0, false_positive=1.0):
    """Deterministic Simpson integration over the random posterior mean."""
    if v == 0:
        return 0.0
    if v == v0:
        p = normal_cdf((mu - threshold) / math.sqrt(v))
        return min(false_negative * p, false_positive * (1.0 - p))
    assert 0 < v < v0
    n = 24000
    left, right = -9.0, 9.0
    h = (right - left) / n
    values = []
    for i in range(n + 1):
        u = left + i * h
        m = mu + math.sqrt(v0 - v) * u
        p = normal_cdf((m - threshold) / math.sqrt(v))
        loss = min(false_negative * p, false_positive * (1 - p))
        density = math.exp(-u * u / 2) / math.sqrt(2 * math.pi)
        weight = 1 if i in (0, n) else (4 if i % 2 else 2)
        values.append(weight * loss * density)
    return math.fsum(values) * h / 3


def exact_counterexample(index):
    states = [(1, 1), (2, 1), (2, 2), (3, 2)]
    groups = defaultdict(list)
    for x1, x2 in states:
        groups[(x1, x2)[index]].append(9 * x1 + 2 * x2)
    variance, risk = Fraction(0), Fraction(0)
    for qs in groups.values():
        probability = Fraction(len(qs), len(states))
        mean = sum(map(Fraction, qs)) / len(qs)
        variance += probability * sum((Fraction(q) - mean)**2 for q in qs) / len(qs)
        p = Fraction(sum(q > 21 for q in qs), len(qs))
        risk += probability * min(9 * p, 1 - p)
    return {"expected_variance": float(variance), "expected_bayes_risk": float(risk)}


def weight_example():
    prior, predicted, q = [0.2, 0.5, 0.3], [0.1, 0.9, 1.5], [0.2, 0.8, 1.4]
    y, sigma = 0.85, 0.25
    logweights = [math.log(w) - 0.5*((y-g)/sigma)**2 for w, g in zip(prior, predicted)]
    top = max(logweights)
    numerators = [math.exp(a-top) for a in logweights]
    weights = [a/sum(numerators) for a in numerators]
    p = sum(w for w, t in zip(weights, q) if t > 1.0)
    return {"weights": weights, "p_exceed": p,
            "no_action_risk": 9*p, "action_risk": 1-p,
            "ess": 1/sum(w*w for w in weights)}


def plugin_risk_expectation(inner_draws, probability=0.1):
    """Exact expectation of min(9*p_hat, 1-p_hat) for binomial p_hat."""
    return sum(math.comb(inner_draws, k) * probability**k *
               (1-probability)**(inner_draws-k) *
               min(9*k/inner_draws, 1-k/inner_draws)
               for k in range(inner_draws + 1))


def main():
    variances = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    risks = [gaussian_expected_risk(v) for v in variances]
    assert all(a <= b + 1e-8 for a, b in zip(risks, risks[1:]))
    x1, x2 = exact_counterexample(0), exact_counterexample(1)
    assert x1 == {"expected_variance": 0.5, "expected_bayes_risk": 0.25}
    assert x2 == {"expected_variance": 20.25, "expected_bayes_risk": 0.0}
    weights = weight_example()
    assert abs(sum(weights["weights"]) - 1.0) < 1e-12
    report = {
        "status": "executed_teaching_examples_not_groundwater_results",
        "gaussian": {"prior_mean": 0, "prior_variance": 1, "threshold": 1,
                     "false_negative_cost": 9, "false_positive_cost": 1,
                     "posterior_variances": variances, "expected_risks": risks,
                     "integration": "Simpson 24000 intervals on [-9,9]; deterministic quadrature"},
        "non_gaussian": {"observe_x1": x1, "observe_x2": x2},
        "weights": weights,
        "jensen": {"true_p": 0.1, "true_risk": 0.9,
                   "expected_plugin_risk": {str(n): plugin_risk_expectation(n) for n in [1, 2, 10, 100]}},
        "combinations": {"24_choose_4": math.comb(24, 4), "12_choose_3": math.comb(12, 3)},
        "all_assertions_passed": True,
    }
    out = Path(__file__).with_name("math_examples.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
