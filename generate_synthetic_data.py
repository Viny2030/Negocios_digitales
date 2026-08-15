import os
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)
OUT = "synthetic_data"
os.makedirs(OUT, exist_ok=True)

CHANNELS = ["email_a", "email_b", "email_c", "retargeting_meta", "sms", "push", "call"]
STAGES = ["awareness", "consideration", "decision"]
n = 400
pd.DataFrame(
    {
        "prospect_id": [f"p{i}" for i in range(n)],
        "channels_contacted": [";".join(random.sample(CHANNELS, k=random.randint(0, 3))) for _ in range(n)],
        "messages_received": np.random.randint(0, 10, n),
        "responses_count": np.random.randint(0, 5, n),
        "days_since_last_contact": np.round(np.random.uniform(0, 30, n), 1),
        "purchase_stage": np.random.choice(STAGES, n, p=[0.5, 0.3, 0.2]),
        "unsubscribed": np.random.choice([True, False], n, p=[0.05, 0.95]),
    }
).to_csv(f"{OUT}/case1_messaging.csv", index=False)

n = 500
pd.DataFrame(
    {
        "product_id": [f"sku-{i}" for i in range(n)],
        "base_cost": np.round(np.random.uniform(10, 500, n), 2),
        "user_segment": np.random.choice(["premium", "standard", "budget"], n, p=[0.3, 0.5, 0.2]),
        "inventory_level_pct": np.round(np.random.uniform(0, 100, n), 1),
        "competitor_avg_price": np.round(np.random.uniform(15, 600, n), 2),
        "demand_index": np.round(np.random.uniform(0, 1, n), 2),
        "min_margin_pct": 0.10,
        "max_multiplier": 2.5,
        "min_multiplier": 1.0,
    }
).to_csv(f"{OUT}/case2_pricing.csv", index=False)

n = 500
is_real = np.random.rand(n) < 0.3
pd.DataFrame(
    {
        "alert_id": [f"a{i}" for i in range(n)],
        "metric_name": np.random.choice(["cpu_usage", "revenue_drop", "latency_p99", "error_rate"], n),
        "z_score": np.where(is_real, np.random.uniform(3, 8, n), np.random.uniform(0, 3, n)),
        "pct_change": np.round(np.random.uniform(-50, 50, n), 1),
        "historical_similar_count": np.random.randint(0, 20, n),
        "estimated_financial_impact": np.round(
            np.where(is_real, np.random.uniform(500, 20000, n), np.random.uniform(0, 500, n)), 2
        ),
        "label": (~is_real).astype(int),
    }
).to_csv(f"{OUT}/case3_anomaly.csv", index=False)

n = 60
pd.DataFrame(
    {
        "startup_id": [f"startup-{i}" for i in range(n)],
        "traction_score": np.round(np.random.uniform(0, 1, n), 2),
        "team_score": np.round(np.random.uniform(0, 1, n), 2),
        "market_score": np.round(np.random.uniform(0, 1, n), 2),
        "runway_months": np.round(np.random.uniform(1, 24, n), 1),
        "past_response_score": np.round(np.random.uniform(0, 1, n), 2),
        "mentorship_hours_received_last_period": np.round(np.random.uniform(0, 40, n), 1),
        "followon_capital_received_last_period": np.round(np.random.uniform(0, 100000, n), 2),
    }
).to_csv(f"{OUT}/case4_accelerator.csv", index=False)

print("CSVs generados en", OUT)
