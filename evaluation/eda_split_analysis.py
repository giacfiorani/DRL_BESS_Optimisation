"""
eda_split_analysis.py — UK Wholesale Electricity Market EDA & Split Recommender
================================================================================
Analyses volatility regimes, price spikes, and extreme events in the BESS
training dataset, then recommends chronological Train/Val/Test splits that
ensure all three partitions contain representative market conditions.

Usage:
    python evalutation/eda_split_analysis.py

Output:
    - 4-panel diagnostic figure saved to results/eda_split_analysis.png
    - Split recommendations printed to stdout with extreme-event counts per bucket
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

# ============================================================
# 1. LOAD & PREP
# ============================================================

PARQUET_PATH = "data/data.parquet"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("BESS Market Data — Exploratory Data Analysis & Split Recommender")
print("=" * 80)

# Load the parquet and set delivery_ts as the datetime index
df = pd.read_parquet(PARQUET_PATH)
df["delivery_ts"] = pd.to_datetime(df["delivery_ts"])
df = df.set_index("delivery_ts").sort_index()

# Ensure numeric price columns (handle any pandas nullable types)
df["da_price"] = pd.to_numeric(df["da_price_gbp_mwh"], errors="coerce").astype(float)
df["id_price"] = pd.to_numeric(df["mid_price_gbp_mwh"], errors="coerce").astype(float)

# DA/ID spread (key arbitrage signal)
df["spread"] = df["da_price"] - df["id_price"]

print(f"\nDataset: {df.index.min().date()} → {df.index.max().date()}")
print(f"Total rows: {len(df):,}  ({len(df)//48:,} delivery days)")
print(f"DA price range: £{df['da_price'].min():.2f} → £{df['da_price'].max():.2f}")
print(f"ID price range: £{df['id_price'].min():.2f} → £{df['id_price'].max():.2f}")

# ============================================================
# 2. VOLATILITY PROFILING
# ============================================================

# Rolling standard deviation of ID prices (the market the agent actually trades on)
# Window sizes: 7-day = 336 half-hours, 30-day = 1440 half-hours
df["id_vol_7d"] = df["id_price"].rolling(window=336, min_periods=168).std()
df["id_vol_30d"] = df["id_price"].rolling(window=1440, min_periods=720).std()

# Rolling volatility of the DA/ID spread
df["spread_vol_7d"] = df["spread"].rolling(window=336, min_periods=168).std()
df["spread_vol_30d"] = df["spread"].rolling(window=1440, min_periods=720).std()

print("\n--- Rolling Volatility Summary (ID Price, 30-day) ---")
vol_yearly = df.groupby(df.index.year)["id_vol_30d"].agg(["mean", "max"])
print(vol_yearly.round(2).to_string())

# ============================================================
# 3. SPIKE / EXTREME EVENT DETECTION
# ============================================================

# Thresholds
P95_DA = df["da_price"].quantile(0.95)
P95_ID = df["id_price"].quantile(0.95)
P99_ID = df["id_price"].quantile(0.99)

print(f"\n--- Price Thresholds ---")
print(f"DA 95th pctile: £{P95_DA:.2f}/MWh")
print(f"ID 95th pctile: £{P95_ID:.2f}/MWh")
print(f"ID 99th pctile: £{P99_ID:.2f}/MWh")

# Flag extreme events
df["is_negative_da"] = df["da_price"] < 0
df["is_negative_id"] = df["id_price"] < 0
df["is_spike_da_95"] = df["da_price"] > P95_DA
df["is_spike_id_95"] = df["id_price"] > P95_ID
df["is_spike_id_99"] = df["id_price"] > P99_ID
df["is_spread_extreme"] = df["spread"].abs() > df["spread"].abs().quantile(0.95)

# Combined "any extreme event" flag
df["is_extreme"] = (
    df["is_negative_da"] | df["is_negative_id"] |
    df["is_spike_da_95"] | df["is_spike_id_95"] |
    df["is_spread_extreme"]
)

# Monthly extreme event density (for the split recommender)
df["year_month"] = df.index.to_period("M")
monthly_extremes = df.groupby("year_month").agg(
    n_steps=("is_extreme", "count"),
    n_extreme=("is_extreme", "sum"),
    n_neg_da=("is_negative_da", "sum"),
    n_neg_id=("is_negative_id", "sum"),
    n_spike_da=("is_spike_da_95", "sum"),
    n_spike_id=("is_spike_id_95", "sum"),
    n_spike_id_99=("is_spike_id_99", "sum"),
    mean_vol_30d=("id_vol_30d", "mean"),
).reset_index()
monthly_extremes["extreme_density"] = monthly_extremes["n_extreme"] / monthly_extremes["n_steps"]

print("\n--- Monthly Extreme Event Summary (top 10 densest months) ---")
print(monthly_extremes.nlargest(10, "extreme_density")[
    ["year_month", "n_extreme", "n_neg_id", "n_spike_id", "extreme_density"]
].to_string(index=False))

# ============================================================
# 4. DATA VISUALIZATION — 4-Panel Diagnostic Figure
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1.2, 1, 1]})
fig.suptitle("UK Wholesale Electricity Market — Volatility & Extreme Event Analysis",
             fontsize=14, fontweight="bold", y=0.98)

# --- Panel 1: Raw Price Timeline ---
ax1 = axes[0]
ax1.plot(df.index, df["da_price"], alpha=0.4, linewidth=0.3, color="#1f77b4", label="DA Price")
ax1.plot(df.index, df["id_price"], alpha=0.4, linewidth=0.3, color="#ff7f0e", label="ID Price")
# Highlight negative prices
neg_mask = df["is_negative_id"]
if neg_mask.any():
    ax1.scatter(df.index[neg_mask], df.loc[neg_mask, "id_price"],
                s=8, c="red", zorder=5, label="Negative ID", alpha=0.7)
ax1.set_ylabel("Price (£/MWh)")
ax1.set_title("Panel A: Raw Price Timeline (DA & ID)")
ax1.legend(loc="upper right", fontsize=8)
ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax1.grid(True, alpha=0.3)

# --- Panel 2: Rolling Volatility ---
ax2 = axes[1]
ax2.plot(df.index, df["id_vol_7d"], linewidth=0.8, color="#2ca02c", alpha=0.6, label="ID Vol (7d)")
ax2.plot(df.index, df["id_vol_30d"], linewidth=1.2, color="#d62728", label="ID Vol (30d)")
ax2.plot(df.index, df["spread_vol_30d"], linewidth=1.0, color="#9467bd",
         linestyle="--", alpha=0.7, label="Spread Vol (30d)")
ax2.set_ylabel("Std Dev (£/MWh)")
ax2.set_title("Panel B: Rolling Volatility (ID Price & DA-ID Spread)")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Extreme Event Scatter (chronological clustering) ---
ax3 = axes[2]
extreme_idx = df.index[df["is_extreme"]]
event_types = []
event_dates = []
event_colors = []
for col, label, color in [
    ("is_negative_id", "Negative ID", "red"),
    ("is_negative_da", "Negative DA", "darkred"),
    ("is_spike_id_95", "ID Spike >P95", "orange"),
    ("is_spike_da_95", "DA Spike >P95", "blue"),
    ("is_spread_extreme", "Spread >P95", "purple"),
]:
    mask = df[col]
    if mask.any():
        ax3.scatter(df.index[mask], [label] * mask.sum(),
                    s=4, alpha=0.5, c=color, label=label)
ax3.set_ylabel("Event Type")
ax3.set_title("Panel C: Extreme Event Clustering (Chronological)")
ax3.legend(loc="upper right", fontsize=7, ncol=3)
ax3.grid(True, alpha=0.3, axis="x")

# --- Panel 4: Monthly Extreme Event Density Bar Chart ---
ax4 = axes[3]
month_dates = [p.to_timestamp() for p in monthly_extremes["year_month"]]
bars = ax4.bar(month_dates, monthly_extremes["extreme_density"],
               width=25, color="#ff7f0e", alpha=0.7, edgecolor="none")
ax4.set_ylabel("Extreme Density\n(fraction of steps)")
ax4.set_title("Panel D: Monthly Extreme Event Density")
ax4.xaxis.set_major_locator(mdates.YearLocator())
ax4.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(OUTPUT_DIR, "eda_split_analysis.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"\nFigure saved → {fig_path}")
plt.close()

# ============================================================
# 5. SPLIT RECOMMENDER
# ============================================================

# The dataset spans delivery dates: 2022-01-01 → 2026-01-01 (exactly 4 years)
data_start = df.index.min().date()
data_end = df.index.max().date()
total_days = (data_end - data_start).days

# Define candidate split regimes (all percentages of total data)
SPLIT_OPTIONS = {
    "Option A — 70/15/15": (0.70, 0.15, 0.15),
    "Option B — 60/20/20": (0.60, 0.20, 0.20),
    "Option C — 75/10/15": (0.75, 0.10, 0.15),
}


def compute_split_dates(train_frac: float, val_frac: float, test_frac: float):
    """Convert fractional splits to exact date boundaries (midnight-aligned)."""
    train_end = data_start + timedelta(days=int(total_days * train_frac))
    val_end = data_start + timedelta(days=int(total_days * (train_frac + val_frac)))
    return train_end, val_end


def count_events_in_range(start_date, end_date):
    """Count extreme events within a date range."""
    mask = (df.index.date >= start_date) & (df.index.date < end_date)
    subset = df.loc[mask]
    n_steps = len(subset)
    if n_steps == 0:
        return {"days": 0, "steps": 0, "neg_da": 0, "neg_id": 0,
                "spike_da_95": 0, "spike_id_95": 0, "spike_id_99": 0,
                "spread_extreme": 0, "total_extreme": 0, "density": 0.0}
    return {
        "days": (end_date - start_date).days,
        "steps": n_steps,
        "neg_da": int(subset["is_negative_da"].sum()),
        "neg_id": int(subset["is_negative_id"].sum()),
        "spike_da_95": int(subset["is_spike_da_95"].sum()),
        "spike_id_95": int(subset["is_spike_id_95"].sum()),
        "spike_id_99": int(subset["is_spike_id_99"].sum()),
        "spread_extreme": int(subset["is_spread_extreme"].sum()),
        "total_extreme": int(subset["is_extreme"].sum()),
        "density": float(subset["is_extreme"].mean()),
    }


print("\n" + "=" * 80)
print("SPLIT RECOMMENDATIONS")
print("=" * 80)

for option_name, (train_f, val_f, test_f) in SPLIT_OPTIONS.items():
    train_end, val_end = compute_split_dates(train_f, val_f, test_f)

    train_stats = count_events_in_range(data_start, train_end)
    val_stats = count_events_in_range(train_end, val_end)
    test_stats = count_events_in_range(val_end, data_end)

    print(f"\n{'─' * 80}")
    print(f"  {option_name}")
    print(f"{'─' * 80}")
    print(f"  Train: {data_start} → {train_end}  ({train_stats['days']} days, {train_stats['steps']:,} steps)")
    print(f"  Val:   {train_end} → {val_end}  ({val_stats['days']} days, {val_stats['steps']:,} steps)")
    print(f"  Test:  {val_end} → {data_end}  ({test_stats['days']} days, {test_stats['steps']:,} steps)")

    # Build the comparison table
    header = f"  {'Metric':<22} {'Train':>10} {'Val':>10} {'Test':>10}"
    print(f"\n{header}")
    print(f"  {'─' * 52}")
    for metric, key in [
        ("Negative DA prices", "neg_da"),
        ("Negative ID prices", "neg_id"),
        ("DA spikes > P95", "spike_da_95"),
        ("ID spikes > P95", "spike_id_95"),
        ("ID spikes > P99", "spike_id_99"),
        ("Spread extremes", "spread_extreme"),
        ("Total extreme events", "total_extreme"),
        ("Extreme density", "density"),
    ]:
        if key == "density":
            print(f"  {metric:<22} {train_stats[key]:>10.4f} {val_stats[key]:>10.4f} {test_stats[key]:>10.4f}")
        else:
            print(f"  {metric:<22} {train_stats[key]:>10,} {val_stats[key]:>10,} {test_stats[key]:>10,}")

    # Warn if any split has zero extreme events of any type
    for split_name, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        missing = []
        if stats["neg_id"] == 0:
            missing.append("negative ID prices")
        if stats["spike_id_95"] == 0:
            missing.append("ID spikes >P95")
        if stats["spread_extreme"] == 0:
            missing.append("extreme spreads")
        if missing:
            print(f"\n  WARNING: {split_name} split has ZERO: {', '.join(missing)}")

# ============================================================
# 6. SUMMARY STATISTICS TABLE
# ============================================================

print(f"\n{'=' * 80}")
print("ANNUAL SUMMARY — Price Statistics by Year")
print(f"{'=' * 80}\n")

annual = df.groupby(df.index.year).agg(
    da_mean=("da_price", "mean"),
    da_std=("da_price", "std"),
    da_min=("da_price", "min"),
    da_max=("da_price", "max"),
    id_mean=("id_price", "mean"),
    id_std=("id_price", "std"),
    id_min=("id_price", "min"),
    id_max=("id_price", "max"),
    n_neg_id=("is_negative_id", "sum"),
    n_spike_id_95=("is_spike_id_95", "sum"),
    spread_mean=("spread", "mean"),
    spread_std=("spread", "std"),
).round(2)

print(annual.to_string())

print(f"\n{'=' * 80}")
print("YOUR CURRENT SPLITS (from CLAUDE.md)")
print(f"{'=' * 80}")
current_train_end = datetime(2024, 10, 19).date()
current_val_end = datetime(2025, 5, 26).date()
cur_train = count_events_in_range(data_start, current_train_end)
cur_val = count_events_in_range(current_train_end, current_val_end)
cur_test = count_events_in_range(current_val_end, data_end)

print(f"\n  Train: {data_start} → {current_train_end}  ({cur_train['days']} days)")
print(f"  Val:   {current_train_end} → {current_val_end}  ({cur_val['days']} days)")
print(f"  Test:  {current_val_end} → {data_end}  ({cur_test['days']} days)")

header = f"\n  {'Metric':<22} {'Train':>10} {'Val':>10} {'Test':>10}"
print(header)
print(f"  {'─' * 52}")
for metric, key in [
    ("Negative DA prices", "neg_da"),
    ("Negative ID prices", "neg_id"),
    ("DA spikes > P95", "spike_da_95"),
    ("ID spikes > P95", "spike_id_95"),
    ("ID spikes > P99", "spike_id_99"),
    ("Spread extremes", "spread_extreme"),
    ("Total extreme events", "total_extreme"),
    ("Extreme density", "density"),
]:
    if key == "density":
        print(f"  {metric:<22} {cur_train[key]:>10.4f} {cur_val[key]:>10.4f} {cur_test[key]:>10.4f}")
    else:
        print(f"  {metric:<22} {cur_train[key]:>10,} {cur_val[key]:>10,} {cur_test[key]:>10,}")

# Warnings for current splits
for split_name, stats in [("Train", cur_train), ("Val", cur_val), ("Test", cur_test)]:
    missing = []
    if stats["neg_id"] == 0:
        missing.append("negative ID prices")
    if stats["spike_id_95"] == 0:
        missing.append("ID spikes >P95")
    if stats["spread_extreme"] == 0:
        missing.append("extreme spreads")
    if missing:
        print(f"\n  WARNING: {split_name} split has ZERO: {', '.join(missing)}")

print(f"\n{'=' * 80}")
print("Done. Review the figure at results/eda_split_analysis.png")
print(f"{'=' * 80}")
