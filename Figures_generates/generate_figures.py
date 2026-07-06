#!/usr/bin/env python3
"""Generate the 5 CINS robustness/influence figures for the paper from the
CSV outputs of InjectionRateSweep.java, IrrelevantNeighborInjectionEpsilonSweep.java,
and NeighborWeightDistribution.java. Run from CARSJava/."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

DATA_DIR = "."
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_NOCINS = "tab:orange"
COLOR_CINS = "tab:blue"


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name + ".png"))
    fig.savefig(os.path.join(OUT_DIR, name + ".pdf"))
    plt.close(fig)
    print("Wrote", os.path.join(OUT_DIR, name + ".{png,pdf}"))


# ---------------------------------------------------------------------------
# Figures 1 & 2 -- injection_rate_sweep.csv
# ---------------------------------------------------------------------------
rate_df = pd.read_csv(os.path.join(DATA_DIR, "injection_rate_sweep.csv"))
rate_df = rate_df.sort_values("rate_pct")

# Figure 1: MAE vs injection rate
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rate_df["rate_pct"], rate_df["MAE_NoCINS"], marker="o", color=COLOR_NOCINS, label="No-CINS (Pearson-kNN)")
ax.plot(rate_df["rate_pct"], rate_df["MAE_CINS"], marker="s", color=COLOR_CINS, label="CINS (FO-CAR)")
ax.set_xlabel("Injection rate (%)")
ax.set_ylabel("MAE")
ax.set_title("Robustness under irrelevant-neighbor injection")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig1_robustness_mae_vs_rate")

# Figure 1b: RMSE variant (same data, alternate metric)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rate_df["rate_pct"], rate_df["RMSE_NoCINS"], marker="o", color=COLOR_NOCINS, label="No-CINS (Pearson-kNN)")
ax.plot(rate_df["rate_pct"], rate_df["RMSE_CINS"], marker="s", color=COLOR_CINS, label="CINS (FO-CAR)")
ax.set_xlabel("Injection rate (%)")
ax.set_ylabel("RMSE")
ax.set_title("Robustness under irrelevant-neighbor injection")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig1b_robustness_rmse_vs_rate")

# Figure 2: Influence Ratio vs injection rate
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rate_df["rate_pct"], rate_df["IR_NoCINS_pct"], marker="o", color=COLOR_NOCINS, label="No-CINS (Pearson-kNN)")
ax.plot(rate_df["rate_pct"], rate_df["IR_CINS_pct"], marker="s", color=COLOR_CINS, label="CINS (FO-CAR)")
ax.set_xlabel("Injection rate (%)")
ax.set_ylabel("Influence Ratio IR (%)")
ax.set_title("Influence Ratio of injected neighbors")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig2_influence_ratio_vs_rate")

# ---------------------------------------------------------------------------
# Figures 3 & 4 -- irrelevant_neighbor_injection_epsilon_sweep.csv
# ---------------------------------------------------------------------------
eps_df = pd.read_csv(os.path.join(DATA_DIR, "irrelevant_neighbor_injection_epsilon_sweep.csv"))
eps_df = eps_df.sort_values("epsilon")

# Figure 3: Elimination rate (left axis) + MAE (right axis) vs EPSILON
fig, ax1 = plt.subplots(figsize=(5, 4))
ax1.plot(eps_df["epsilon"], eps_df["elim_pct"], marker="o", color="tab:blue", label="Elimination rate (%)")
ax1.set_xlabel(r"$\epsilon$ (Choquet threshold)")
ax1.set_ylabel("Elimination rate (%)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(eps_df["epsilon"], eps_df["MAE_inject_CINS"], marker="s", color="tab:red", label="MAE (+50% injection)")
ax2.set_ylabel("MAE", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title(r"Effect of $\epsilon$: selectivity vs. accuracy trade-off")
ax1.grid(alpha=0.3)
fig.tight_layout()
save(fig, "fig3_epsilon_elimination_vs_mae")

# Figure 4: Influence Ratio vs EPSILON (CINS curve + No-CINS reference line)
fig, ax = plt.subplots(figsize=(5, 4))
ir_nocins_ref = eps_df["IR_NoCINS_pct"].iloc[0]
ax.axhline(ir_nocins_ref, color=COLOR_NOCINS, linestyle="--", label=f"No-CINS (reference, {ir_nocins_ref:.1f}%)")
ax.plot(eps_df["epsilon"], eps_df["IR_CINS_pct"], marker="o", color=COLOR_CINS, label="CINS")
ax.set_xlabel(r"$\epsilon$ (Choquet threshold)")
ax.set_ylabel("Influence Ratio IR (%)")
ax.set_title(r"Influence Ratio vs. $\epsilon$")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig4_influence_ratio_vs_epsilon")

# ---------------------------------------------------------------------------
# Figure 5 -- neighbor_weight_distribution.csv
# ---------------------------------------------------------------------------
w_df = pd.read_csv(os.path.join(DATA_DIR, "neighbor_weight_distribution.csv"))
genuine = w_df[w_df["group"] == "genuine"]["final_weight"]
injected = w_df[w_df["group"] == "injected"]["final_weight"]

# Figure 5a: histogram
fig, ax = plt.subplots(figsize=(5, 4))
bins = 40
ax.hist(genuine, bins=bins, alpha=0.6, density=True, color=COLOR_CINS, label=f"Genuine neighbors (n={len(genuine)})")
ax.hist(injected, bins=bins, alpha=0.6, density=True, color=COLOR_NOCINS, label=f"Injected neighbors (n={len(injected)})")
ax.set_xlabel("Final Choquet weight (pearson x choquet)")
ax.set_ylabel("Density")
ax.set_title("Distribution of neighbor weights: genuine vs. injected")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig5a_weight_histogram")

# Figure 5b: violin plot (alternate visualization)
fig, ax = plt.subplots(figsize=(5, 4))
parts = ax.violinplot([genuine.values, injected.values], showmedians=True, showextrema=True)
for pc, color in zip(parts["bodies"], [COLOR_CINS, COLOR_NOCINS]):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Genuine", "Injected"])
ax.set_ylabel("Final Choquet weight (pearson x choquet)")
ax.set_title("Distribution of neighbor weights: genuine vs. injected")
ax.grid(alpha=0.3)
save(fig, "fig5b_weight_violin")

print("\nSummary stats (final_weight):")
print(w_df.groupby("group")["final_weight"].describe())
print("\nKept fraction (choquet >= EPSILON):")
print(w_df.groupby("group")["kept"].mean())

print("\nAll figures written to", os.path.abspath(OUT_DIR))
