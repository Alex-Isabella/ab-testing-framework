"""
ab_tests.py
-----------
Reusable A/B testing framework functions for marketing and product analysts.
Covers pre-experiment planning, data validation, statistical testing, and
business recommendation generation.

Author: Alex Isabella
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.stats.power as smp
from statsmodels.stats.proportion import proportion_effectsize as _proportion_effectsize
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PHASE 1: SAMPLE SIZE & POWER CALCULATIONS
# ─────────────────────────────────────────────

def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> dict:
    """
    Calculate required sample size per variant before running an experiment.

    Parameters
    ----------
    baseline_rate : float
        Current conversion rate (e.g., 0.08 for 8%).
    mde : float
        Minimum detectable effect as absolute change (e.g., 0.02 for +2pp lift).
    alpha : float
        Significance level. Default 0.05.
    power : float
        Statistical power (1 - beta). Default 0.80.
    two_tailed : bool
        Use two-tailed test. Default True.

    Returns
    -------
    dict with sample size, expected lift %, and runtime estimate.
    """
    target_rate = baseline_rate + mde
    effect_size = _proportion_effectsize(baseline_rate, target_rate)
    ratio = 1 if two_tailed else None

    analysis = smp.NormalIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative="two-sided" if two_tailed else "larger"
    )
    n = int(np.ceil(n))
    relative_lift = (target_rate - baseline_rate) / baseline_rate * 100

    return {
        "sample_size_per_variant": n,
        "total_sample_size": n * 2,
        "baseline_rate": baseline_rate,
        "target_rate": target_rate,
        "absolute_mde": mde,
        "relative_lift_pct": round(relative_lift, 2),
        "alpha": alpha,
        "power": power,
        "two_tailed": two_tailed,
    }


def estimate_runtime_days(sample_size_per_variant: int, daily_traffic: int) -> dict:
    """
    Estimate how many days the experiment needs to run.

    Parameters
    ----------
    sample_size_per_variant : int
    daily_traffic : int
        Total daily visitors split across both variants.

    Returns
    -------
    dict with runtime estimates and warnings.
    """
    daily_per_variant = daily_traffic / 2
    days_needed = np.ceil(sample_size_per_variant / daily_per_variant)

    warnings_list = []
    if days_needed < 7:
        warnings_list.append("⚠️ Runtime < 7 days — risk of novelty effect. Consider running at least 1 full week.")
    if days_needed > 60:
        warnings_list.append("⚠️ Runtime > 60 days — consider increasing MDE or accepting higher alpha.")

    return {
        "days_needed": int(days_needed),
        "weeks_needed": round(days_needed / 7, 1),
        "daily_traffic": daily_traffic,
        "daily_per_variant": int(daily_per_variant),
        "warnings": warnings_list,
    }


# ─────────────────────────────────────────────
# PHASE 2: DATA VALIDATION & SANITY CHECKS
# ─────────────────────────────────────────────

def check_sample_ratio_mismatch(
    n_control: int,
    n_variant: int,
    expected_split: float = 0.5,
    alpha: float = 0.01
) -> dict:
    """
    Check for Sample Ratio Mismatch (SRM) — a sign of a broken randomization
    or data pipeline. Uses a chi-square test against the expected split.

    Parameters
    ----------
    n_control : int
    n_variant : int
    expected_split : float
        Expected fraction in each group. Default 0.5 (50/50).
    alpha : float
        SRM significance threshold. Default 0.01 (stricter than experiment alpha).

    Returns
    -------
    dict with chi-square result and SRM flag.
    """
    total = n_control + n_variant
    expected_control = total * expected_split
    expected_variant = total * (1 - expected_split)

    chi2, p_value = stats.chisquare(
        f_obs=[n_control, n_variant],
        f_exp=[expected_control, expected_variant]
    )

    srm_detected = p_value < alpha
    actual_split = n_control / total

    return {
        "n_control": n_control,
        "n_variant": n_variant,
        "actual_split_control": round(actual_split, 4),
        "expected_split": expected_split,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "srm_detected": srm_detected,
        "interpretation": (
            "🚨 SRM DETECTED — Do NOT trust experiment results. "
            "Investigate randomization and data pipeline before proceeding."
            if srm_detected
            else "✅ No SRM detected. Sample split looks healthy."
        ),
    }


def check_normality(data: np.ndarray, group_name: str = "group") -> dict:
    """
    Shapiro-Wilk normality test. Use on samples < 5000;
    for larger samples, rely on CLT or use Kolmogorov-Smirnov.

    Parameters
    ----------
    data : array-like
    group_name : str

    Returns
    -------
    dict with test result and recommendation.
    """
    n = len(data)

    if n > 5000:
        # For large samples, Shapiro-Wilk loses sensitivity — use KS test
        ks_stat, p_value = stats.kstest(data, "norm",
                                         args=(np.mean(data), np.std(data)))
        test_used = "Kolmogorov-Smirnov"
        statistic = ks_stat
    else:
        statistic, p_value = stats.shapiro(data)
        test_used = "Shapiro-Wilk"

    is_normal = p_value > 0.05

    return {
        "group": group_name,
        "n": n,
        "test_used": test_used,
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "is_normal": is_normal,
        "interpretation": (
            f"✅ {group_name}: Data appears normally distributed (p={p_value:.4f})."
            if is_normal
            else f"⚠️ {group_name}: Data is NOT normally distributed (p={p_value:.4f}). "
                 "Consider Mann-Whitney U or bootstrap."
        ),
    }


def check_variance_equality(control: np.ndarray, variant: np.ndarray) -> dict:
    """
    Levene's test for equality of variances.
    Determines whether to use Student's t-test or Welch's t-test.
    """
    stat, p_value = stats.levene(control, variant)
    equal_variance = p_value > 0.05

    return {
        "levene_statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "equal_variance": equal_variance,
        "recommended_test": "Student's t-test" if equal_variance else "Welch's t-test",
        "interpretation": (
            "✅ Variances are equal — Student's t-test appropriate."
            if equal_variance
            else "⚠️ Unequal variances detected — use Welch's t-test."
        ),
    }


def detect_novelty_effect(
    daily_control: pd.Series,
    daily_variant: pd.Series,
    warmup_days: int = 3
) -> dict:
    """
    Compare early-period vs steady-state conversion rates to flag novelty effects.
    A novelty effect occurs when variant lifts are disproportionately high at launch
    then decay as users adjust to the change.

    Parameters
    ----------
    daily_control : pd.Series
        Daily conversion rate for control over time.
    daily_variant : pd.Series
        Daily conversion rate for variant over time.
    warmup_days : int
        Days to treat as "early period." Default 3.

    Returns
    -------
    dict with novelty effect assessment.
    """
    if len(daily_variant) <= warmup_days:
        return {"error": "Not enough days of data to assess novelty effect."}

    early_lift = (daily_variant.iloc[:warmup_days].mean() -
                  daily_control.iloc[:warmup_days].mean())
    steady_lift = (daily_variant.iloc[warmup_days:].mean() -
                   daily_control.iloc[warmup_days:].mean())

    decay_pct = ((early_lift - steady_lift) / abs(early_lift) * 100
                 if early_lift != 0 else 0)

    novelty_flagged = decay_pct > 30  # >30% lift decay after warmup = flag

    return {
        "warmup_days": warmup_days,
        "early_period_lift": round(float(early_lift), 6),
        "steady_state_lift": round(float(steady_lift), 6),
        "lift_decay_pct": round(float(decay_pct), 2),
        "novelty_effect_flagged": novelty_flagged,
        "interpretation": (
            f"🚨 Novelty effect detected — lift decayed {decay_pct:.1f}% after warmup period. "
            "Steady-state lift is the more reliable signal."
            if novelty_flagged
            else f"✅ No strong novelty effect. Lift appears stable post-warmup."
        ),
    }


# ─────────────────────────────────────────────
# PHASE 3: SIGNIFICANCE TESTING
# ─────────────────────────────────────────────

def run_proportions_ztest(
    control_conversions: int,
    control_visitors: int,
    variant_conversions: int,
    variant_visitors: int,
    alpha: float = 0.05
) -> dict:
    """
    Two-proportion z-test for binary metrics (conversion rate, CTR, phone call rate).
    Best used when metric is a proportion and sample is large (n > 30 per group).
    """
    from statsmodels.stats.proportion import proportions_ztest, proportion_confint

    count = np.array([variant_conversions, control_conversions])
    nobs = np.array([variant_visitors, control_visitors])

    z_stat, p_value = proportions_ztest(count, nobs)

    control_rate = control_conversions / control_visitors
    variant_rate = variant_conversions / variant_visitors
    absolute_lift = variant_rate - control_rate
    relative_lift = absolute_lift / control_rate * 100

    # 95% confidence intervals
    ci_control = proportion_confint(control_conversions, control_visitors, alpha=alpha)
    ci_variant = proportion_confint(variant_conversions, variant_visitors, alpha=alpha)

    is_significant = p_value < alpha

    return {
        "test": "Two-Proportion Z-Test",
        "control_rate": round(control_rate, 6),
        "variant_rate": round(variant_rate, 6),
        "absolute_lift": round(absolute_lift, 6),
        "relative_lift_pct": round(relative_lift, 2),
        "z_statistic": round(float(z_stat), 4),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "is_significant": is_significant,
        "ci_control": (round(ci_control[0], 6), round(ci_control[1], 6)),
        "ci_variant": (round(ci_variant[0], 6), round(ci_variant[1], 6)),
        "interpretation": (
            f"✅ Statistically significant. Variant CVR ({variant_rate:.2%}) vs "
            f"Control ({control_rate:.2%}). Relative lift: {relative_lift:.1f}% (p={p_value:.4f})."
            if is_significant
            else f"❌ Not statistically significant (p={p_value:.4f}). Cannot conclude "
                 f"the variant outperforms control."
        ),
    }


def run_ttest(
    control: np.ndarray,
    variant: np.ndarray,
    equal_var: bool = False,
    alpha: float = 0.05
) -> dict:
    """
    Welch's t-test (default) or Student's t-test for continuous metrics
    (e.g., revenue per user, session duration, AOV).

    Parameters
    ----------
    control : array-like
    variant : array-like
    equal_var : bool
        False = Welch's t-test (default, safer). True = Student's t-test.
    alpha : float
    """
    test_name = "Student's t-test" if equal_var else "Welch's t-test"
    t_stat, p_value = stats.ttest_ind(variant, control, equal_var=equal_var)

    control_mean = np.mean(control)
    variant_mean = np.mean(variant)
    absolute_lift = variant_mean - control_mean
    relative_lift = absolute_lift / control_mean * 100

    # Confidence interval for the difference
    se = np.sqrt(np.var(variant, ddof=1) / len(variant) +
                 np.var(control, ddof=1) / len(control))
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(variant) + len(control) - 2)
    ci = (absolute_lift - t_crit * se, absolute_lift + t_crit * se)

    is_significant = p_value < alpha

    return {
        "test": test_name,
        "control_mean": round(float(control_mean), 4),
        "variant_mean": round(float(variant_mean), 4),
        "absolute_lift": round(float(absolute_lift), 4),
        "relative_lift_pct": round(float(relative_lift), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "is_significant": is_significant,
        "ci_95_diff": (round(ci[0], 4), round(ci[1], 4)),
        "interpretation": (
            f"✅ Statistically significant. Variant mean ({variant_mean:.4f}) vs "
            f"Control ({control_mean:.4f}). Relative lift: {relative_lift:.1f}% (p={p_value:.4f})."
            if is_significant
            else f"❌ Not statistically significant (p={p_value:.4f})."
        ),
    }


def run_mann_whitney(
    control: np.ndarray,
    variant: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Mann-Whitney U test — non-parametric alternative to t-test.
    Use when data is not normally distributed (e.g., revenue with heavy right-skew).
    """
    u_stat, p_value = stats.mannwhitneyu(variant, control, alternative="two-sided")

    control_median = np.median(control)
    variant_median = np.median(variant)
    median_lift = variant_median - control_median
    median_lift_pct = median_lift / control_median * 100

    is_significant = p_value < alpha

    return {
        "test": "Mann-Whitney U Test",
        "control_median": round(float(control_median), 4),
        "variant_median": round(float(variant_median), 4),
        "median_lift": round(float(median_lift), 4),
        "median_lift_pct": round(float(median_lift_pct), 2),
        "u_statistic": round(float(u_stat), 4),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "is_significant": is_significant,
        "note": "Reports median lift (not mean) — appropriate for skewed distributions.",
        "interpretation": (
            f"✅ Statistically significant (p={p_value:.4f}). Variant median "
            f"({variant_median:.4f}) vs Control ({control_median:.4f})."
            if is_significant
            else f"❌ Not statistically significant (p={p_value:.4f})."
        ),
    }


def run_bootstrap(
    control: np.ndarray,
    variant: np.ndarray,
    n_iterations: int = 10000,
    alpha: float = 0.05,
    metric_func=np.mean
) -> dict:
    """
    Bootstrap hypothesis test — distribution-free and works for any metric
    (mean, median, revenue per user, etc.). Most robust for small or skewed samples.

    Parameters
    ----------
    control : array-like
    variant : array-like
    n_iterations : int
        Bootstrap resamples. 10,000 is standard.
    alpha : float
    metric_func : callable
        Aggregation function (np.mean, np.median, etc.)
    """
    np.random.seed(42)
    observed_diff = metric_func(variant) - metric_func(control)

    boot_diffs = []
    for _ in range(n_iterations):
        boot_control = np.random.choice(control, size=len(control), replace=True)
        boot_variant = np.random.choice(variant, size=len(variant), replace=True)
        boot_diffs.append(metric_func(boot_variant) - metric_func(boot_control))

    boot_diffs = np.array(boot_diffs)

    ci_lower = np.percentile(boot_diffs, alpha / 2 * 100)
    ci_upper = np.percentile(boot_diffs, (1 - alpha / 2) * 100)
    p_value = np.mean(boot_diffs <= 0) * 2  # two-tailed

    is_significant = not (ci_lower <= 0 <= ci_upper)
    relative_lift = observed_diff / abs(metric_func(control)) * 100

    return {
        "test": "Bootstrap (Permutation)",
        "n_iterations": n_iterations,
        "observed_diff": round(float(observed_diff), 6),
        "relative_lift_pct": round(float(relative_lift), 2),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "is_significant": is_significant,
        "interpretation": (
            f"✅ Bootstrap CI [{ci_lower:.4f}, {ci_upper:.4f}] excludes 0 — "
            f"statistically significant at {alpha} level."
            if is_significant
            else f"❌ Bootstrap CI [{ci_lower:.4f}, {ci_upper:.4f}] includes 0 — "
                 f"not statistically significant."
        ),
    }


def select_and_run_test(
    control: np.ndarray = None,
    variant: np.ndarray = None,
    metric_type: str = "continuous",
    control_conversions: int = None,
    control_visitors: int = None,
    variant_conversions: int = None,
    variant_visitors: int = None,
    alpha: float = 0.05
) -> dict:
    """
    Auto-selects the appropriate statistical test based on metric type
    and data distribution.

    Decision logic:
        - Binary metric (proportion) → Z-test
        - Continuous + normal + equal variance → Student's t-test
        - Continuous + normal + unequal variance → Welch's t-test
        - Continuous + non-normal → Mann-Whitney U
        - Small sample (n < 30 per group) → Bootstrap

    Parameters
    ----------
    metric_type : str
        "proportion" or "continuous"
    """
    decision_log = []

    if metric_type == "proportion":
        decision_log.append("Metric type: proportion → running Two-Proportion Z-Test.")
        result = run_proportions_ztest(
            control_conversions, control_visitors,
            variant_conversions, variant_visitors,
            alpha=alpha
        )
        result["decision_log"] = decision_log
        return result

    # Continuous metric: check sample size
    n_min = min(len(control), len(variant))
    if n_min < 30:
        decision_log.append(f"Small sample (n={n_min} < 30) → running Bootstrap.")
        result = run_bootstrap(control, variant, alpha=alpha)
        result["decision_log"] = decision_log
        return result

    # Normality check
    norm_control = check_normality(control, "control")
    norm_variant = check_normality(variant, "variant")
    both_normal = norm_control["is_normal"] and norm_variant["is_normal"]

    if not both_normal:
        decision_log.append("Non-normal distribution detected → running Mann-Whitney U.")
        result = run_mann_whitney(control, variant, alpha=alpha)
        result["decision_log"] = decision_log
        return result

    # Variance check
    var_check = check_variance_equality(control, variant)
    equal_var = var_check["equal_variance"]

    if equal_var:
        decision_log.append("Normal + equal variance → running Student's t-test.")
    else:
        decision_log.append("Normal + unequal variance → running Welch's t-test.")

    result = run_ttest(control, variant, equal_var=equal_var, alpha=alpha)
    result["decision_log"] = decision_log
    return result


# ─────────────────────────────────────────────
# PHASE 4: BUSINESS RECOMMENDATION
# ─────────────────────────────────────────────

def assess_practical_significance(
    relative_lift_pct: float,
    min_business_lift_pct: float = 2.0
) -> dict:
    """
    Statistical significance ≠ practical significance.
    A 0.1% lift can be statistically significant with large enough traffic
    but meaningless to the business.

    Parameters
    ----------
    relative_lift_pct : float
        Observed relative lift %.
    min_business_lift_pct : float
        Minimum lift the business considers meaningful. Default 2%.
    """
    is_practical = abs(relative_lift_pct) >= min_business_lift_pct

    return {
        "observed_lift_pct": relative_lift_pct,
        "minimum_business_lift_pct": min_business_lift_pct,
        "is_practically_significant": is_practical,
        "interpretation": (
            f"✅ Lift of {relative_lift_pct:.1f}% exceeds business threshold of {min_business_lift_pct}%."
            if is_practical
            else f"⚠️ Lift of {relative_lift_pct:.1f}% is below business threshold of {min_business_lift_pct}%. "
                 "Statistically significant but may not be worth shipping."
        ),
    }


def generate_recommendation(
    test_result: dict,
    srm_result: dict = None,
    novelty_result: dict = None,
    practical_result: dict = None,
    experiment_name: str = "Unnamed Experiment",
    hypothesis: str = "",
    analyst_notes: str = ""
) -> str:
    """
    Generate a plain-English business recommendation memo from test results.
    Designed to be copy-pasteable for stakeholder communication.

    Returns
    -------
    str : Formatted markdown memo.
    """
    is_sig = test_result.get("is_significant", False)
    lift = test_result.get("relative_lift_pct", test_result.get("median_lift_pct", 0))
    p_val = test_result.get("p_value", None)
    test_name = test_result.get("test", "Statistical Test")

    # Determine recommendation
    caveats = []

    if srm_result and srm_result.get("srm_detected"):
        recommendation = "⛔ DO NOT SHIP — EXPERIMENT INVALIDATED"
        caveats.append("Sample Ratio Mismatch detected. Results are unreliable.")
    elif novelty_result and novelty_result.get("novelty_effect_flagged"):
        recommendation = "⏳ EXTEND EXPERIMENT"
        caveats.append(
            f"Novelty effect flagged (lift decayed {novelty_result['lift_decay_pct']:.1f}% "
            "after warmup). Run longer to observe steady-state behavior."
        )
    elif is_sig and (practical_result is None or practical_result.get("is_practically_significant")):
        recommendation = "✅ SHIP IT"
    elif is_sig and practical_result and not practical_result.get("is_practically_significant"):
        recommendation = "⚠️ SHIP WITH CAUTION"
        caveats.append(
            f"Statistically significant but lift ({lift:.1f}%) is below business threshold "
            f"({practical_result['minimum_business_lift_pct']}%). Evaluate implementation cost."
        )
    else:
        recommendation = "❌ DO NOT SHIP"
        caveats.append("Results are not statistically significant. No evidence of a real effect.")

    # Build memo
    memo = f"""
# A/B Test Analysis Report
**Experiment:** {experiment_name}
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Analyst:** Alex Isabella

---

## Hypothesis
{hypothesis if hypothesis else '_No hypothesis provided._'}

---

## Results Summary

| Metric | Value |
|--------|-------|
| Statistical Test Used | {test_name} |
| Relative Lift | {lift:+.2f}% |
| P-Value | {f'{p_val:.4f}' if p_val is not None else 'N/A'} |
| Statistically Significant | {'Yes ✅' if is_sig else 'No ❌'} |
| Practically Significant | {'Yes ✅' if (practical_result and practical_result.get('is_practically_significant')) else ('N/A' if practical_result is None else 'No ⚠️')} |

---

## Recommendation

### {recommendation}

{'**Caveats:**' if caveats else ''}
{''.join(f'{chr(10)}- {c}' for c in caveats)}

---

## Interpretation
{test_result.get('interpretation', '')}

{'**Novelty Effect:** ' + novelty_result.get('interpretation', '') if novelty_result else ''}
{'**SRM Check:** ' + srm_result.get('interpretation', '') if srm_result else ''}

---

## Analyst Notes
{analyst_notes if analyst_notes else '_No additional notes._'}

---

## Decision Log (Test Selection)
{chr(10).join(f'- {step}' for step in test_result.get('decision_log', ['Manual test selection.']))}

---
_Report generated by ab-testing-framework | github.com/AlexIsabella_
""".strip()

    return memo
