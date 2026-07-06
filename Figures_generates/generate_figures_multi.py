#!/usr/bin/env python3
"""Generate the 5 CINS robustness/influence figures for RichMTV, LDOS, and
Food, from the CSV outputs of *_Multi.java (LDOS/Food) and the original
RichMTV-only Java files (different filenames, handled below). Run from
CARSJava/."""

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
COLOR_NOCINS = "tab:orange"
COLOR_CINS = "tab:blue"

# Maps dataset name -> (rate_sweep_csv, epsilon_sweep_csv, weight_dist_csv)
DATASETS = {
    "RichMTV": ("injection_rate_sweep.csv", "irrelevant_neighbor_injection_epsilon_sweep.csv", "neighbor_weight_distribution.csv"),
    "LDOS":    ("injection_rate_sweep_LDOS.csv", "epsilon_sweep_LDOS.csv", "neighbor_weight_distribution_LDOS.csv"),
    "Food":    ("injection_rate_sweep_Food.csv", "epsilon_sweep_Food.csv", "neighbor_weight_distribution_Food.csv"),
    "RichML":  ("injection_rate_sweep_RichML.csv", "epsilon_sweep_RichML.csv", "neighbor_weight_distribution_RichML.csv"),
    "InCar":   ("injection_rate_sweep_InCar.csv", "epsilon_sweep_InCar.csv", "neighbor_weight_distribution_InCar.csv"),
}


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name + ".png"))
    fig.savefig(os.path.join(out_dir, name + ".pdf"))
    plt.close(fig)
    print("Wrote", os.path.join(out_dir, name + ".{png,pdf}"))


def make_figures(dataset, rate_csv, eps_csv, weight_csv):
    out_dir = os.path.join("figures", dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {dataset} ===")

    # -- Figures 1 & 2 --
    rate_path = os.path.join(DATA_DIR, rate_csv)
    if not os.path.exists(rate_path):
        print(f"  SKIP rate-sweep figures ({rate_path} not found)")
    else:
        rate_df = pd.read_csv(rate_path).sort_values("rate_pct")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rate_df["rate_pct"], rate_df["MAE_NoCINS"], marker="o", color=COLOR_NOCINS, label="No-CINS (Pearson-kNN)")
        ax.plot(rate_df["rate_pct"], rate_df["MAE_CINS"], marker="s", color=COLOR_CINS, label="CINS (FO-CAR)")
        ax.set_xlabel("Injection rate (%)")
        ax.set_ylabel("MAE")
        ax.set_title(f"Robustness under irrelevant-neighbor injection ({dataset})")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, out_dir, "fig1_robustness_mae_vs_rate")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rate_df["rate_pct"], rate_df["RMSE_NoCINS"], marker="o", color=COLOR_NOCINS, label="No-CINS (Pearson-kNN)")
        ax.plot(rate_df["rate_pct"], rate_df["RMSE_CINS"], marker="s", color=COLOR_CINS, label="CINS (FO-CAR)")
        ax.set_xlabel("Injection rate (%)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Robustness under irrelevant-neighbor injection ({dataset})")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, out_dir, "fig1b_robustness_rmse_vs_rate")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rate_df["rate_pct"], rate_df["IR_NoCINS_pct"], marker="o", color=COLOR_NOCINS, label="No-CINS (Pearson-kNN)")
        ax.plot(rate_df["rate_pct"], rate_df["IR_CINS_pct"], marker="s", color=COLOR_CINS, label="CINS (FO-CAR)")
        ax.set_xlabel("Injection rate (%)")
        ax.set_ylabel("Influence Ratio IR (%)")
        ax.set_title(f"Influence Ratio of injected neighbors ({dataset})")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, out_dir, "fig2_influence_ratio_vs_rate")

    # -- Figures 3 & 4 --
    eps_path = os.path.join(DATA_DIR, eps_csv)
    if not os.path.exists(eps_path):
        print(f"  SKIP epsilon-sweep figures ({eps_path} not found)")
    else:
        eps_df = pd.read_csv(eps_path).sort_values("epsilon")

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
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left" if dataset != "Food" else "center left")
        ax1.set_title(rf"Effect of $\epsilon$: selectivity vs. accuracy trade-off ({dataset})")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        save(fig, out_dir, "fig3_epsilon_elimination_vs_mae")

        fig, ax = plt.subplots(figsize=(5, 4))
        ir_nocins_ref = eps_df["IR_NoCINS_pct"].iloc[0]
        ax.axhline(ir_nocins_ref, color=COLOR_NOCINS, linestyle="--", label=f"No-CINS (reference, {ir_nocins_ref:.1f}%)")
        ax.plot(eps_df["epsilon"], eps_df["IR_CINS_pct"], marker="o", color=COLOR_CINS, label="CINS")
        ax.set_xlabel(r"$\epsilon$ (Choquet threshold)")
        ax.set_ylabel("Influence Ratio IR (%)")
        ax.set_title(rf"Influence Ratio vs. $\epsilon$ ({dataset})")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, out_dir, "fig4_influence_ratio_vs_epsilon")

    # -- Figure 5 --
    weight_path = os.path.join(DATA_DIR, weight_csv)
    if not os.path.exists(weight_path):
        print(f"  SKIP weight-distribution figures ({weight_path} not found)")
    else:
        w_df = pd.read_csv(weight_path)
        genuine = w_df[w_df["group"] == "genuine"]["final_weight"]
        injected = w_df[w_df["group"] == "injected"]["final_weight"]

        fig, ax = plt.subplots(figsize=(5, 4))
        bins = 40
        ax.hist(genuine, bins=bins, alpha=0.6, density=True, color=COLOR_CINS, label=f"Genuine neighbors (n={len(genuine)})")
        ax.hist(injected, bins=bins, alpha=0.6, density=True, color=COLOR_NOCINS, label=f"Injected neighbors (n={len(injected)})")
        ax.set_xlabel("Final Choquet weight (pearson x choquet)")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of neighbor weights ({dataset})")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, out_dir, "fig5a_weight_histogram")

        fig, ax = plt.subplots(figsize=(5, 4))
        parts = ax.violinplot([genuine.values, injected.values], showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], [COLOR_CINS, COLOR_NOCINS]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Genuine", "Injected"])
        ax.set_ylabel("Final Choquet weight (pearson x choquet)")
        ax.set_title(f"Distribution of neighbor weights ({dataset})")
        ax.grid(alpha=0.3)
        save(fig, out_dir, "fig5b_weight_violin")

        print(f"  {dataset} weight stats:")
        print(w_df.groupby("group")["final_weight"].describe())
        print(f"  {dataset} kept fraction:")
        print(w_df.groupby("group")["kept"].mean())


if __name__ == "__main__":
    for dataset, (rate_csv, eps_csv, weight_csv) in DATASETS.items():
        make_figures(dataset, rate_csv, eps_csv, weight_csv)
    print("\nAll figures written under figures/<dataset>/")
