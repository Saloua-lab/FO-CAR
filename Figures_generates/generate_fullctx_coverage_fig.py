import matplotlib.pyplot as plt

# FullCtx (r^C) exact-all-dims-match coverage, measured empirically per dataset
# (one 80/20 split, seed=1, first fold) via FullCtxCoverage_<Dataset>.java
data = [
    ("Music", 0.00),
    ("LDOS-CoMoDa", 5.01),
    ("RichMovieLens", 8.97),
    ("RichMTV", 28.76),
    ("Movie", 68.25),
    ("Food", 100.00),
]
names = [d[0] for d in data]
values = [d[1] for d in data]

fig, ax = plt.subplots(figsize=(8.5, 4.5))
bars = ax.bar(names, values, color="#4C72B0", width=0.6)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.2f}%",
            ha="center", va="bottom", fontsize=10)

plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

ax.set_ylabel("FullCtx coverage (%)")
ax.set_ylim(0, 108)
ax.set_title("FullCtx (r$^C$) exact all-dimensions match coverage by dataset")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("figures/fullctx_coverage_by_dataset.png", dpi=200)
plt.savefig("figures/fullctx_coverage_by_dataset.pdf")
print("Saved figures/fullctx_coverage_by_dataset.png and .pdf")
