"""
generate_data.py
----------------
Generates three synthetic experiment datasets representing real-world scenarios:
  1. Ship it    — clear, statistically and practically significant winner
  2. Don't ship — no significant difference between control and variant
  3. Inconclusive — significant but tiny lift; novelty effect present

Run this script once to populate the /data directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def make_daily_rates(base_rate, variant_rate, n_days, novelty_boost=0.0, warmup_days=3):
    control_daily, variant_daily = [], []
    for d in range(n_days):
        boost = novelty_boost if d < warmup_days else 0.0
        control_daily.append(np.random.normal(base_rate, base_rate * 0.05))
        variant_daily.append(np.random.normal(variant_rate + boost, variant_rate * 0.05))
    return pd.Series(control_daily), pd.Series(variant_daily)


# ─── SCENARIO 1: SHIP IT ─────────────────────────────────────────────────────
# New CTA button test on landing page. Variant drives meaningful CVR lift.
n_visitors = 6000
control_cvr, variant_cvr = 0.080, 0.097   # +21% relative lift

control_c = np.random.binomial(1, control_cvr, n_visitors)
variant_c = np.random.binomial(1, variant_cvr, n_visitors)

scenario1 = pd.DataFrame({
    "user_id": range(n_visitors * 2),
    "group": ["control"] * n_visitors + ["variant"] * n_visitors,
    "converted": list(control_c) + list(variant_c),
    "session_duration_sec": (
        list(np.random.lognormal(5.5, 0.6, n_visitors)) +
        list(np.random.lognormal(5.7, 0.6, n_visitors))
    ),
    "day": list(np.random.randint(1, 22, n_visitors)) * 2,
})
scenario1.to_csv(DATA_DIR / "scenario1_ship_it.csv", index=False)

# Daily rates for novelty check
ctrl_daily_1, var_daily_1 = make_daily_rates(control_cvr, variant_cvr, 21)
pd.DataFrame({"control_rate": ctrl_daily_1, "variant_rate": var_daily_1}).to_csv(
    DATA_DIR / "scenario1_daily_rates.csv", index=False
)
print("✅ Scenario 1 saved: scenario1_ship_it.csv")


# ─── SCENARIO 2: DO NOT SHIP ─────────────────────────────────────────────────
# Homepage headline copy test. No meaningful difference.
n_visitors2 = 5000
control_cvr2, variant_cvr2 = 0.055, 0.057   # ~3.6% relative, not significant

control_c2 = np.random.binomial(1, control_cvr2, n_visitors2)
variant_c2 = np.random.binomial(1, variant_cvr2, n_visitors2)

scenario2 = pd.DataFrame({
    "user_id": range(n_visitors2 * 2),
    "group": ["control"] * n_visitors2 + ["variant"] * n_visitors2,
    "converted": list(control_c2) + list(variant_c2),
    "revenue_usd": (
        list(np.random.lognormal(3.8, 1.2, n_visitors2)) +
        list(np.random.lognormal(3.81, 1.2, n_visitors2))
    ),
    "day": list(np.random.randint(1, 29, n_visitors2)) * 2,
})
scenario2.to_csv(DATA_DIR / "scenario2_do_not_ship.csv", index=False)

ctrl_daily_2, var_daily_2 = make_daily_rates(control_cvr2, variant_cvr2, 28)
pd.DataFrame({"control_rate": ctrl_daily_2, "variant_rate": var_daily_2}).to_csv(
    DATA_DIR / "scenario2_daily_rates.csv", index=False
)
print("✅ Scenario 2 saved: scenario2_do_not_ship.csv")


# ─── SCENARIO 3: INCONCLUSIVE (NOVELTY EFFECT) ────────────────────────────────
# Redesigned form layout. Early spike in conversions that fades — classic novelty.
n_visitors3 = 8000
control_cvr3, variant_cvr3 = 0.065, 0.068   # Tiny real lift

control_c3 = np.random.binomial(1, control_cvr3, n_visitors3)
variant_c3 = np.random.binomial(1, variant_cvr3, n_visitors3)

# Inject novelty boost into first 3 days of variant raw data
days3 = np.random.randint(1, 22, n_visitors3)
novelty_boost_mask = days3 <= 3
variant_c3_adj = variant_c3.copy()
variant_c3_adj[novelty_boost_mask] = np.random.binomial(
    1, 0.11, novelty_boost_mask.sum()
)

scenario3 = pd.DataFrame({
    "user_id": range(n_visitors3 * 2),
    "group": ["control"] * n_visitors3 + ["variant"] * n_visitors3,
    "converted": list(control_c3) + list(variant_c3_adj),
    "revenue_usd": (
        list(np.random.lognormal(4.1, 1.5, n_visitors3)) +
        list(np.random.lognormal(4.12, 1.5, n_visitors3))
    ),
    "day": list(days3) * 2,
})
scenario3.to_csv(DATA_DIR / "scenario3_inconclusive.csv", index=False)

ctrl_daily_3, var_daily_3 = make_daily_rates(
    control_cvr3, variant_cvr3, 21, novelty_boost=0.045
)
pd.DataFrame({"control_rate": ctrl_daily_3, "variant_rate": var_daily_3}).to_csv(
    DATA_DIR / "scenario3_daily_rates.csv", index=False
)
print("✅ Scenario 3 saved: scenario3_inconclusive.csv")

print("\n✅ All datasets generated in /data directory.")
