# src/detection/optimization/grid_search.py
import sys
import os
import csv
import yaml
import glob
import copy
import json
import itertools
from datetime import datetime
import matplotlib.pyplot as plt

from inference.detect import process_one_video

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def set_cfg(cfg: dict, dotted: str, val):
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def write_csv(path, keys, records):
    """Writes out a header of keys + metrics, then one row per record."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["loss", "det_rate", "avg_props"])
        for params, loss, det, ap in records:
            w.writerow([params[k] for k in keys] + [loss, det, ap])


def plot_losses(path, losses):
    """Scatter‐plot Loss vs. combo index."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.scatter(range(1, len(losses) + 1), losses)
    plt.xlabel("Combo index")
    plt.ylabel("Loss")
    plt.title("Grid Search Loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def log_run(log_path, config_path, param_grid, results_csv, plot_path):
    """
    Append a record of this grid-search run to a master CSV.
    Columns: timestamp, config, param_grid (JSON), results_csv, plot_path.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        fieldnames = ["timestamp", "config", "param_grid", "results_csv", "plot_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp":    datetime.now().isoformat(),
            "config":       config_path,
            "param_grid":   json.dumps(param_grid),
            "results_csv":  results_csv,
            "plot_path":    plot_path
        })

        
def run_grid_search(config_path: str, top_k: int = 10):
    # load base config + pull grid
    base = yaml.safe_load(open(config_path))
    grid = base.pop("param_grid", {})

    # expand video list
    inp = base["paths"]["input_video"]
    if os.path.isdir(inp):
        videos = sorted(sum((glob.glob(os.path.join(inp, ext)) for ext in VIDEO_EXTS), []))
    else:
        videos = [inp]

    if not videos:
        print(f"[ERROR] No videos found in {inp}")
        return

    keys, lists = zip(*grid.items())
    combos = list(itertools.product(*lists))
    results = []

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"\n[GRID] Combo {idx}/{len(combos)} → {params}")

        # inject params
        cfg = copy.deepcopy(base)
        for k, v in params.items():
            set_cfg(cfg, k, v)

        # run inference on each video
        ap_list, dr_list = [], []
        for vp in videos:
            ap, dr = process_one_video(vp, cfg)
            ap_list.append(ap)
            dr_list.append(dr)

        avg_props     = sum(ap_list) / len(ap_list)
        overall_det   = sum(dr_list) / len(dr_list)
        α, β          = cfg["scoring"]["alpha"], cfg["scoring"]["beta"]
        loss_value    = -α * overall_det + β * avg_props

        # per‐combo terminal info
        print(f" → avg_props: {avg_props:.2f}, det_rate: {overall_det:.1%}, loss: {loss_value:.3f}")

        results.append((params, loss_value, overall_det, avg_props))

    # write CSV + plot
    out_dir   = base.get("paths", {}).get("opt_output_dir", "output/optimization")
    csv_path  = os.path.join(out_dir, "grid_search_results.csv")
    plot_path = os.path.join(out_dir, "grid_search_loss.png")

    write_csv(csv_path, list(keys), results)
    print(f"\n[INFO] CSV results written to {csv_path}")

    plot_losses(plot_path, [r[1] for r in results])
    print(f"[INFO] Loss plot saved to {plot_path}")

    # top‐K summary
    sorted_idx = sorted(range(len(results)), key=lambda i: results[i][1])[:top_k]
    print(f"\nTop {top_k}:")
    for rank, i in enumerate(sorted_idx, start=1):
        p, l, d, a = results[i]
        print(f" {rank}. idx={i+1} loss={l:.3f}, det={d:.1%}, props={a:.1f} params={p}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml"
    run_grid_search(cfg)
