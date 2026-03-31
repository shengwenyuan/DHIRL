import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# CONFIG    = os.path.join(ROOT, "src_autotest/configs/test0318.yaml")
CONFIG    = os.path.join(ROOT, "src_autotest/configs/test0331.yaml")
# RUN_DIR   = os.path.join(ROOT, "src_autotest/outputs/20260318_193739")
RUN_DIR   = os.path.join(ROOT, "src_autotest/outputs/")
OUT_DIR   = os.path.join(ROOT, "outputs/labyrinth_train")
GROUP     = "big_table"
X_PARAM   = "num_latents"
X_LABEL   = "K"
GROUP_BY  = "model_type"
LL_FILE   = "ll.csv"
OUT       = os.path.join(OUT_DIR, f"hiddenK_linechart.pdf")
COLORS    = ["#e4c40e", "#196eab", "#4dbd4d"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def resolve_experiment(defaults, experiment):
    params = dict(defaults)
    params.update(experiment)
    return params

def find_group(cfg, group_name):
    groups = cfg.get("groups", {})
    if group_name not in groups:
        raise KeyError(f"Group '{group_name}' not found. Available: {list(groups.keys())}")
    group = groups[group_name]
    defaults = cfg.get("defaults", {})
    experiments = [resolve_experiment(defaults, exp) for exp in group["experiments"]]
    return group["id"], experiments

def read_test_ll(csv_path):
    return pd.read_csv(csv_path)["test_ll"].values


cfg = load_config(CONFIG)
group_id, experiments = find_group(cfg, GROUP)

raw = defaultdict(list)
for exp in experiments:
    exp_id  = exp["id"]
    x_val   = exp.get(X_PARAM)
    grp_val = exp.get(GROUP_BY)
    if x_val is None or grp_val is None:
        continue
    csv_path = os.path.join(RUN_DIR, group_id, exp_id, LL_FILE)
    if not os.path.exists(csv_path):
        print(f"[warn] {exp_id}: not found at {csv_path}")
        continue
    lls = read_test_ll(csv_path)
    raw[grp_val].append((x_val, lls.mean(), lls.std()))

for key in raw:
    raw[key].sort(key=lambda t: t[0])

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
for i, (grp_val, points) in enumerate(sorted(raw.items())):
    c     = COLORS[i % len(COLORS)]
    xs    = [p[0] for p in points]
    means = [p[1] for p in points]
    stds  = [p[2] for p in points]
    ax.plot(xs, means, marker="o", color=c, linewidth=2, label=str(grp_val)[9:])
    ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], color=c, alpha=0.12)

ax.set_xlabel(X_LABEL, fontsize=18)
ax.set_ylabel("Test LL", fontsize=18)
ax.tick_params(axis="both", labelsize=13)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

if all(isinstance(x, int) for grp in raw.values() for x, _, _ in grp):
    ax.set_xticks(sorted({x for grp in raw.values() for x, _, _ in grp}))

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
plt.savefig(OUT)
print(f"Saved to {OUT}")
