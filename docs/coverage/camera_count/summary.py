"""Summarise the camera-count study: reads N_cameras/layout_plan.json for each N and
writes coverage_by_camera_count.png and summary.json next to this file, and prints the
tables used in README.md.

    python docs/coverage/camera_count/summary.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = [("whole room", "Whole room"), ("2,2,8,10", "Work area (2, 2)-(8, 10)"),
         ("0,0.5,4,2", "Bottom-left corner (0, 0.5)-(4, 2)")]
POSTURES = ("seated", "standing")
# Reference palette (dataviz skill), validated for these two series on the light surface.
COLOURS = {"seated": "#2a78d6", "standing": "#eb6834"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def pct(s, key):
    return 100.0 * s[key] / s["floor_m2"] if s["floor_m2"] else 0.0


def load():
    rows = []
    for path in glob.glob(os.path.join(HERE, "*_cameras", "layout_plan.json")):
        n = int(re.match(r"(\d+)_cameras", os.path.basename(os.path.dirname(path))).group(1))
        with open(path) as f:
            plan = json.load(f)
        best = plan["layouts"][0]
        rows.append({"cameras": n, "score": best["score"], "zones": best["zones"],
                     "agreement": plan.get("agreement"), "best_seed": plan.get("best_seed"),
                     "runs": [{"seed": r["seed"], "score": r["score"],
                               "largest_move_from_best_m": r.get("largest_move_from_best_m")}
                              for r in plan.get("runs", [])],
                     "layout": best["cameras"]})
    return sorted(rows, key=lambda r: r["cameras"])


def chart(rows, path):
    ns = [r["cameras"] for r in rows]
    fig, axes = plt.subplots(1, len(ZONES), figsize=(11, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes, ZONES):
        ax.set_facecolor(SURFACE)
        ends = {}
        for p in POSTURES:
            ys = [pct(r["zones"][key][p], "good_m2") for r in rows]
            ax.plot(ns, ys, color=COLOURS[p], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=p.capitalize(), zorder=3)
            ends[p] = ys[-1]
        # End labels; when the two lines finish close together, one label says both.
        if abs(ends["seated"] - ends["standing"]) < 4:
            y = (ends["seated"] + ends["standing"]) / 2
            text = (f"{ends['seated']:.0f}%" if round(ends["seated"]) == round(ends["standing"])
                    else f"{ends['standing']:.0f}% / {ends['seated']:.0f}%")
            ax.annotate(text, (ns[-1], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        else:
            for p in POSTURES:
                ax.annotate(f"{ends[p]:.0f}%", (ns[-1], ends[p]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(ns)
        ax.set_xlim(min(ns) - 0.3, max(ns) + 0.6)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.set_xlabel("Number of cameras", fontsize=9, color=MUTED)
    axes[0].set_ylabel("Good floor (%)", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.99, 1.0), labelcolor=INK)
    fig.suptitle("Predicted good floor by number of cameras (best of 10 runs each)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def main():
    rows = load()
    chart(rows, os.path.join(HERE, "coverage_by_camera_count.png"))
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("| Cameras | " + " | ".join(t for _, t in ZONES) + " | Unseen, whole room | Gain in good floor |")
    print("|---|" + "---|" * (len(ZONES) + 2))
    prev = None
    for r in rows:
        cells = [" / ".join(f"{pct(r['zones'][k][p], 'good_m2'):.0f}" for p in POSTURES) for k, _ in ZONES]
        unseen = " / ".join(f"{pct(r['zones']['whole room'][p], 'unseen_m2'):.0f}" for p in POSTURES)
        good = [pct(r["zones"]["whole room"][p], "good_m2") for p in POSTURES]
        gain = "" if prev is None else " / ".join(f"{g - q:+.0f}" for g, q in zip(good, prev))
        prev = good
        print(f"| {r['cameras']} | " + " | ".join(cells) + f" | {unseen} | {gain or 'n/a'} |")

    print("\n| Cameras | Runs within 0.5% of the best | Score range | Largest camera move between runs |")
    print("|---|---|---|---|")
    for r in rows:
        a = r["agreement"]
        moves = [x["largest_move_from_best_m"] for x in r["runs"] if x["largest_move_from_best_m"] is not None]
        print(f"| {r['cameras']} | {a['within_half_percent_of_best']} of {a['runs']} | "
              f"{a['score_min']:.1f} to {a['score_max']:.1f} | {max(moves):.2f} m |")

    for r in rows:
        print(f"\n**{r['cameras']} cameras** (best run: seed {r['best_seed']})\n")
        print("| Camera | Mount on line | Position x, y (m) | Height (m) | Face towards x, y | Tilt down |")
        print("|---|---|---|---|---|---|")
        for c in r["layout"]:
            x, y, z = c["position_m"]
            fx, fy = c["face_towards_m"]
            print(f"| {c['label']} | L{c.get('mount_line', '?')} | {x:.2f}, {y:.2f} | {z:.2f} | "
                  f"{fx:.2f}, {fy:.2f} | {c['tilt_down_deg']:.0f}° |")


if __name__ == "__main__":
    main()
