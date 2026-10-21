"""
make_thesis_tables.py
LaTeX tables (booktabs tabulars, no caption: the captions go in the thesis) and one CSV per table, plus the coverage
bar chart, into viz_env/thesis_tables/ of the data checkout. READ-ONLY: it reads the existing per-target evaluation
npz files, the eval / analyze logs, the env target cache and the cached BEV predictions; no model is loaded, nothing
is trained or evaluated, and the test split is not re-run (its eval_env_test_*.npz / *_test_final.log are only read).

  1 env_per_scene_<split>     comparisons (1), (2), (3), (3b) of runs/DECISION_RULES_ENV.md: per-scene seed means
                              with / without, the lower one bold ("with" lower = a win of the rule), ALL seed mean
                              +/- std, wins, gap vs 2 x max std, verdict. Recomputed from runs/E_*/eval_env_<split>_
                              checkpoint_epoch12.npz with analyze_env.py's functions; every per-scene value and verdict
                              is checked against analyze_env.log / analyze_env_final_test.log (|diff| <= 5e-4).
  2 env_subsets_<split>       car, two_wheeler, car_moving, two_wheeler_moving, stationary, moving: n, minADE6, ADE1,
                              miss rate (MR1: FDE1 > 2 m, MR6: minFDE6 > 2 m) for E_EgoOnly / E_V2X / E_V2Xnomap, seed
                              mean +/- std (sample std over the 3 seeds; descriptive). E_EgoOnly predicts class (a)
                              only, so its n are smaller. Rows also in the logs are checked against them.
  3 env_coverage              per split and town (and ALL): kept targets by visibility class ego (a) / coop CAV only
                              (b) / RSU only / nobody (the two parts of (c), cache rsu_vis), all and moving (cache
                              speed > 0), counts and shares, over every D sample of the split's scenes (eval_env.py's
                              coverage rows). Window-filtered vehicles (cache n_removed) are a separate column (not in
                              the shares). Checked against the npz metas' coverage counts and the npz moving counts.
                              + env_coverage.pdf: stacked bars of the shares, 16 cm wide, text >= 8 pt, the BEV
                              figures' class colours (render_env_bev.CLASS_COLORS).
  4 ego_per_scene_<split>     ego task (Part 1, split v2, mirror, epoch 12) minADE6 per scene, seed means of C_D,
                              C_Dmapv2, C_Dnomap, C_EgoOnly from eval_<run>_e12.log / eval_<run>_test_final.log; best
                              per scene bold; ALL seed mean +/- std.
  5 town04_30_failure         tuning Town04 #30, E_V2X on moving (b) targets: n targets, distinct vehicles (cache
                              inst), share of two-wheelers, speed distribution (m/s), ADE1 model (seed mean +/- std)
                              vs constant velocity per target type; the other 7 tuning scenes' moving (b) targets as a
                              reference column. Top-1 overshoot / undershoot (top-1 path length - ground-truth path
                              length over the valid steps, > +2 m / < -2 m): only the stored top-1 paths exist, i.e.
                              viz_env/pred/E_V2X_seed0_scene30_ego3.npz (seed 0, ego CAV 3's samples; verified against
                              the eval npz there by render_env_bev.load_preds). This subset is printed with its n.

Run in gpu-test-venv from the data checkout:  python ~/hd-maps-dev/make_thesis_tables.py
"""
import os
import re
import csv
import json
import statistics as st

import numpy as np

from analyze_env import load, check_runs, RULES, GROUPS, EXPECT, METRIC_NAME
from env_metrics import subset_masks, scene_values, rule, MISS_THRESHOLD_M
from env_dataset import load_env_cache, ENV_CACHE_DIR
from render_env_bev import CLASS_COLORS, FRAME_RATE_HZ

OUT = "viz_env/thesis_tables"
SEEDS = (0, 1, 2)
NPZ = "runs/E_{g}_seed{s}/eval_env_{split}_checkpoint_epoch12.npz"
ENV_LOG = {"tuning": "analyze_env.log", "test": "analyze_env_final_test.log"}
EGO_MODELS = ("C_D", "C_Dmapv2", "C_Dnomap", "C_EgoOnly")
EGO_LOG = {"tuning": "eval_{run}_e12.log", "test": "eval_{run}_test_final.log"}
SUBSETS2 = ("car", "two_wheeler", "car_moving", "two_wheeler_moving", "stationary", "moving")
SUBSET_TEX = {"car": "car", "two_wheeler": "two-wheeler", "car_moving": "car, moving",
              "two_wheeler_moving": "two-wheeler, moving", "stationary": "stationary", "moving": "moving"}
GROUP_TEX = {"EgoOnly": "E\\_EgoOnly", "V2X": "E\\_V2X", "V2Xnomap": "E\\_V2Xnomap"}
RULE_TEX = {"(1) V2X on (a)": "(1) V2X on (a)", "(2) V2X on (b) vs constant velocity": "(2) V2X on (b) vs CV",
            "(3) map, moving targets": "(3) map, moving", "(3b) map, turning targets (secondary)": "(3b) map, turning"}
METRIC_TEX = {"minade6": "minADE$_6$", "ade1": "ADE$_1$", "cv_ade": "CV ADE"}
CLASSES = ("ego", "coop CAV only", "RSU only", "nobody")
SHOOT_M = 2.0
TOL = 5e-4


def fmt(x, d=3):
    return f"{x:.{d}f}"


def pm(vals, d=3):
    return f"{st.mean(vals):.{d}f} $\\pm$ {st.stdev(vals):.{d}f}", f"{st.mean(vals):.{d}f} +/- {st.stdev(vals):.{d}f}"


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


def write(name, header_tex, colspec, rows_tex, rows_csv, csv_header, rules_after=()):
    """header_tex: list of header lines (already ending in \\\\); rows_tex: body lines; rules_after: body line indices
    followed by a \\midrule."""
    path = os.path.join(OUT, f"{name}.tex")
    with open(path, "w") as f:
        f.write(f"% {name}: generated by make_thesis_tables.py (read-only; needs \\usepackage{{booktabs}})\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        f.write("\n".join(header_tex) + "\n\\midrule\n")
        for i, r in enumerate(rows_tex):
            f.write(r + "\n")
            if i in rules_after:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)
        w.writerows(rows_csv)
    return path


def show(title, header, rows):
    """Plain-text table on stdout."""
    cells = [header] + [[str(c) for c in r] for r in rows]
    wid = [max(len(r[i]) for r in cells) for i in range(len(header))]
    print(f"\n== {title}")
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, wid))))
        if j == 0:
            print("  " + "  ".join("-" * w for w in wid))


# ---------------------------------------------------------------- env npz
def env_runs(split):
    runs, metas = {}, {}
    for g in GROUPS:
        for s in SEEDS:
            name = f"E_{g}_seed{s}"
            runs[name], metas[name] = load(NPZ.format(g=g, s=s, split=split))
            assert (metas[name]["v2x"], metas[name]["no_map"]) == EXPECT[g], name
            assert metas[name]["ckpt"].endswith("checkpoint_epoch12.pt"), name
    by = check_runs(runs, metas, split)
    G = {g: by[f"E_{g}"] for g in GROUPS}
    scenes = next(iter(metas.values()))["scenes"]
    return G, metas, scenes


def log_rules(split):
    """analyze_env log -> {rule label: ({scene: (with, without)}, verdict)}."""
    txt = open(ENV_LOG[split]).read()
    out = {}
    for label, *_ in RULES:
        i = txt.index(f"\n{label}:")
        block = txt[i:txt.index("\n\n", i + 1)]
        per = {int(s): (float(a), float(b)) for s, a, b in re.findall(r"#(\d+): ([\d.]+)/([\d.]+)", block)}
        verdict = re.search(rf"\n  {re.escape(label)}\s+(HELPS / better|no detectable effect)", txt).group(1)
        out[label] = (per, verdict)
    return out


def log_subsets(split):
    """analyze_env log 'Seed mean +/- std' rows -> {(group, subset): {metric: (mean, std)}}."""
    out = {}
    for g, sub, rest in re.findall(r"\n  (EgoOnly|V2X|V2Xnomap)\s+(\w+)\s+(minade6=.*)", open(ENV_LOG[split]).read()):
        out[(g, sub)] = {k: (float(a), float(b)) for k, a, b in re.findall(r"(\w+)=\s*([\d.]+)\+/-([\d.]+)", rest)}
    return out


# ---------------------------------------------------------------- 1
def table1(split, G, scenes):
    ref = log_rules(split)
    sc = [s for _, s in scenes]
    town = {s: t for t, s in scenes}
    res = {}
    for label, gw, mw, go, mo, sub in RULES:
        w = [scene_values(t, mw, subset_masks(t)[sub], sc) for t in G[gw]]
        o = [scene_values(t, mo, subset_masks(t)[sub], sc) for t in G[go]]
        r = rule(w, o, sc)
        per = {s: (st.mean(x[s] for x in w), st.mean(x[s] for x in o)) for s in sc}
        lp, lv = ref[label]
        for s in sc:
            assert max(abs(per[s][0] - lp[s][0]), abs(per[s][1] - lp[s][1])) <= TOL, (split, label, s, per[s], lp[s])
        verdict = "HELPS / better" if r["helps"] else "no detectable effect"
        assert verdict == lv, (split, label, verdict, lv)
        res[label] = (per, r, [x["ALL"] for x in w], [x["ALL"] for x in o], verdict)
    head = ["scene"] + [f"{RULE_TEX[l]} {k}" for l, *_ in RULES for k in ("with", "w/o")]
    rows_txt, rows_tex, rows_csv = [], [], []
    for s in sc:
        cells_txt, cells_tex = [f"{town[s]} #{s}"], [f"{town[s]} \\#{s}"]
        for label, *_ in RULES:
            a, b = res[label][0][s]
            cells_txt += [fmt(a) + ("*" if a < b else ""), fmt(b) + ("*" if b < a else "")]
            cells_tex += [bold(fmt(a), a < b), bold(fmt(b), b < a)]
            rows_csv.append([split, label, town[s], s, round(a, 4), round(b, 4), int(a < b)])
        rows_txt.append(cells_txt)
        rows_tex.append(" & ".join(cells_tex) + " \\\\")
    summ_txt = {"ALL (mean +/- std)": [], "wins (with lower)": [], "gap / 2 x max std": [], "verdict": []}
    summ_tex = {"ALL": [], "wins": [], "gap / thr.": [], "verdict": []}
    for label, *_ in RULES:
        per, r, aw, ao, verdict = res[label]
        tw, xw = pm(aw)
        to, xo = pm(ao) if r["sd_without"] > 0 else (fmt(r["without"]), fmt(r["without"]))
        summ_txt["ALL (mean +/- std)"] += [xw, xo]
        summ_tex["ALL"] += [tw, to]
        summ_txt["wins (with lower)"] += [f"{r['wins']}/8", ""]
        summ_tex["wins"] += [f"\\multicolumn{{2}}{{c}}{{{r['wins']}/8}}"]
        summ_txt["gap / 2 x max std"] += [f"{r['gap']:+.3f} / {r['threshold']:.3f}", ""]
        summ_tex["gap / thr."] += [f"\\multicolumn{{2}}{{c}}{{{r['gap']:+.3f} / {r['threshold']:.3f}}}"]
        v = "helps" if r["helps"] else "no effect"
        summ_txt["verdict"] += [v, ""]
        summ_tex["verdict"] += [f"\\multicolumn{{2}}{{c}}{{{v}}}"]
        rows_csv.append([split, label, "ALL", "ALL", round(r["with"], 4), round(r["without"], 4), int(r["helps"]),
                         r["wins"], round(r["gap"], 4), round(r["threshold"], 4), round(r["sd_with"], 4),
                         round(r["sd_without"], 4), verdict])
    n_body = len(rows_tex)
    rows_txt += [[k] + v for k, v in summ_txt.items()]
    rows_tex += [" & ".join([k] + v) + " \\\\" for k, v in summ_tex.items()]
    header_tex = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{RULE_TEX[l]}}}" for l, *_ in RULES) + " \\\\",
                  "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(RULES))),
                  "scene & " + " & ".join(f"{METRIC_TEX[mw]} & {METRIC_TEX[mo]}" for _, _, mw, _, mo, _ in RULES)
                  + " \\\\",
                  " & " + " & ".join(f"{GROUP_TEX[gw]} & {GROUP_TEX[go] if mo != 'cv_ade' else 'CV'}"
                                     for _, gw, _, go, mo, _ in RULES) + " \\\\"]
    write(f"env_per_scene_{split}", header_tex, "l" + "rr" * len(RULES), rows_tex, rows_csv,
          ["split", "comparison", "town", "scene", "with", "without", "with_lower_or_helps", "wins", "gap",
           "threshold", "sd_with", "sd_without", "verdict"], rules_after=(n_body - 1,))
    show(f"1 env per scene, {split} (m; * = lower of the pair; with = first column of each pair; "
         + "; ".join(f"{l}: {gw} {METRIC_NAME[mw]} vs {go} {METRIC_NAME[mo]} on {sub}" for l, gw, mw, go, mo, sub
                     in RULES) + ")", head, rows_txt)


# ---------------------------------------------------------------- 2
def table2(split, G):
    ref = log_subsets(split)
    keys = ("minade6", "ade1", "miss1", "miss6")
    head = ["subset", "model", "n", "minADE6 (m)", "ADE1 (m)", "MR1", "MR6"]
    rows_txt, rows_tex, rows_csv, n_checked = [], [], [], 0
    for sub in SUBSETS2:
        for gi, g in enumerate(GROUPS):
            ms = [subset_masks(t)[sub] for t in G[g]]
            n = int(ms[0].sum())
            assert all(int(m.sum()) == n for m in ms)
            vals = {k: [float(t[k][m].mean()) for t, m in zip(G[g], ms)] for k in keys}
            for k in keys:
                if (g, sub) in ref:
                    a, b = ref[(g, sub)][k]
                    assert abs(st.mean(vals[k]) - a) <= TOL and abs(st.stdev(vals[k]) - b) <= TOL, (split, g, sub, k)
                    n_checked += 1
            cells = [pm(vals[k]) for k in keys]
            rows_txt.append([SUBSET_TEX[sub] if gi == 0 else "", g, n] + [c[1] for c in cells])
            rows_tex.append(" & ".join([SUBSET_TEX[sub] if gi == 0 else "", GROUP_TEX[g], str(n)]
                                       + [c[0] for c in cells]) + " \\\\")
            rows_csv.append([split, sub, g, n] + [round(x, 4) for k in keys for x in (st.mean(vals[k]),
                                                                                     st.stdev(vals[k]))])
    header_tex = ["subset & model & $n$ & minADE$_6$ (m) & ADE$_1$ (m) & MR$_1$ & MR$_6$ \\\\"]
    write(f"env_subsets_{split}", header_tex, "llrrrrr", rows_tex, rows_csv,
          ["split", "subset", "model", "n", "minade6_mean", "minade6_std", "ade1_mean", "ade1_std", "mr1_mean",
           "mr1_std", "mr6_mean", "mr6_std"], rules_after=tuple(3 * i + 2 for i in range(len(SUBSETS2) - 1)))
    show(f"2 env by subset, {split} (seed mean +/- std over 3 seeds; MR = miss rate, final error > "
         f"{MISS_THRESHOLD_M:g} m; {n_checked} values checked against {ENV_LOG[split]})", head, rows_txt)


# ---------------------------------------------------------------- 3
def coverage_counts(cache, scenes):
    """{town or 'ALL': {'all': [4], 'moving': [4], 'removed': n, 'samples': n}} over every D sample of the scenes."""
    ptr = cache["sample_ptr"]
    out = {}
    for tw in sorted({t for t, _ in scenes}) + ["ALL"]:
        sc = [s for t, s in scenes if tw in (t, "ALL")]
        rows = np.nonzero(np.isin(cache["sample_scene"], sc))[0]
        idx = np.concatenate([np.arange(ptr[r], ptr[r + 1]) for r in rows])
        vis, rsu, mov = cache["vis"][idx], cache["rsu_vis"][idx], cache["speed"][idx] > 0
        cls = np.where(vis <= 1, vis, np.where(rsu, 2, 3))
        out[tw] = {"all": np.bincount(cls, minlength=4), "moving": np.bincount(cls[mov], minlength=4),
                   "removed": int(cache["n_removed"][rows].sum()), "samples": len(rows)}
    return out


def table3(cache, splits):
    cov = {}
    for split, (G, metas, scenes) in splits.items():
        cov[split] = coverage_counts(cache, scenes)
        c, m = cov[split]["ALL"], metas[f"E_V2X_seed0"]["coverage"]
        assert (int(c["all"][0]), int(c["all"][1]), int(c["all"][2] + c["all"][3]), c["removed"], c["samples"]) == \
            (m["a"], m["b"], m["c"], m["removed"], m["samples"]), (split, c, m)
        ref = G["V2X"][0]
        assert int(c["moving"][0]) == int(((ref["vis"] == 0) & ref["moving"]).sum())
        assert int(c["moving"][1]) == int(((ref["vis"] == 1) & ref["moving"]).sum())
        r = metas["E_V2XRSU_seed0"]["coverage"]["r"] if "E_V2XRSU_seed0" in metas else None
        if r is not None:
            assert r == int(c["all"][2]), (split, r, c["all"][2])
    head = ["split", "town", "targets", "n"] + [f"{k} %" for k in CLASSES] + ["window-filtered"]
    rows_txt, rows_tex, rows_csv = [], [], []
    for split in cov:
        for tw, c in cov[split].items():
            for kind in ("all", "moving"):
                v = c[kind]
                n = int(v.sum())
                sh = 100 * v / n
                rem = str(c["removed"]) if kind == "all" else ""
                rows_txt.append([split, tw, kind, n] + [f"{x:.1f} ({k})" for x, k in zip(sh, v)] + [rem])
                rows_tex.append(" & ".join([split if (tw, kind) == (next(iter(cov[split])), "all") else "",
                                            tw if kind == "all" else "", kind, str(n)]
                                           + [f"{x:.1f}" for x in sh] + [rem]) + " \\\\")
                rows_csv.append([split, tw, kind, n] + [int(k) for k in v] + [round(x, 2) for x in sh]
                                + [c["removed"] if kind == "all" else ""])
    header_tex = ["split & town & targets & $n$ & ego (\\%) & coop CAV only (\\%) & RSU only (\\%) & nobody (\\%) & "
                  "window-filt. \\\\"]
    n_tun = 2 * len(cov["tuning"])
    write("env_coverage", header_tex, "lllrrrrrr", rows_tex, rows_csv,
          ["split", "town", "targets", "n", "n_ego", "n_coop_only", "n_rsu_only", "n_nobody", "pct_ego",
           "pct_coop_only", "pct_rsu_only", "pct_nobody", "window_filtered"], rules_after=(n_tun - 1,))
    show("3 coverage by visibility class (kept targets of every D sample of the split's scenes; % of n, counts in "
         "brackets; window-filtered vehicles not in n)", head, rows_txt)
    coverage_chart(cov)


def coverage_chart(cov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    matplotlib.rcParams["pdf.fonttype"] = 42
    labels = [(split, tw) for split in cov for tw in cov[split]]
    W, H = 16 / 2.54, 0.34 + 0.2 * len(labels) + 0.8
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    y = np.arange(len(labels))[::-1].astype(float)
    y[:len(cov["tuning"])] += 0.5                          # gap between the splits
    for ax, kind in zip(axes, ("all", "moving")):
        left = np.zeros(len(labels))
        for k, (name, col) in enumerate(zip(CLASSES, CLASS_COLORS)):
            sh = np.array([100 * cov[s][t][kind][k] / cov[s][t][kind].sum() for s, t in labels])
            ax.barh(y, sh, left=left, height=0.72, color=col, edgecolor="white", lw=0.4, label=name)
            for yi, l, v in zip(y, left, sh):
                if v >= 7:
                    ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if k != 1 else "black")
            left += sh
        for yi, (s, t) in zip(y, labels):
            ax.text(101, yi, f"n={cov[s][t][kind].sum()}", ha="left", va="center", fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=8, length=2)
        ax.set_xlabel("share of kept targets (%)", fontsize=8)
        ax.set_title("all targets" if kind == "all" else "moving targets", fontsize=9, loc="left", pad=3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y)
    axes[1].tick_params(axis="y", length=0)
    axes[0].set_yticklabels([f"{t} ({s})" for s, t in labels], fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    lg = fig.legend(h, ["seen by the ego", "coop CAV only", "RSU only", "seen by nobody"], loc="lower center",
                    ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.2,
                    columnspacing=1.0)
    fig.subplots_adjust(left=0.17, right=0.9, top=1 - 0.3 / H, bottom=0.72 / H, wspace=0.34)
    fig.canvas.draw()
    bb = lg.get_window_extent().transformed(fig.transFigure.inverted())
    xl = max(a.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).y0 for a in axes)
    assert bb.x0 >= 0 and bb.x1 <= 1 and bb.y0 >= 0 and bb.y1 <= xl, f"legend does not fit: {bb}"
    small = {t.get_text(): t.get_fontsize() for t in fig.findobj(Text)
             if t.get_visible() and t.get_text() and t.get_fontsize() < 8}
    assert not small, f"text below 8 pt: {small}"
    path = os.path.join(OUT, "env_coverage.pdf")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path} ({W * 2.54:.0f} x {H * 2.54:.1f} cm)")


# ---------------------------------------------------------------- 4
EGO_ROW = re.compile(r"^(Town0\d)\s+(\d+)\s+(\d+)\s+([\d.]+)m")
EGO_ALL = re.compile(r"^ALL\s+(\d+)\s+([\d.]+)m")


def ego_log(path, split):
    txt = open(path).read()
    assert f"Evaluating on the {split.upper()} split" in txt and "map_transform=mirror, split_version=v2" in txt, path
    assert re.search(r"ckpt=runs/\S+/checkpoint_epoch12\.pt", txt), path
    per, tot = {}, None
    for line in txt.splitlines():
        if m := EGO_ROW.match(line):
            per[(m.group(1), int(m.group(2)))] = float(m.group(4))
        elif m := EGO_ALL.match(line):
            tot = float(m.group(2))
    assert len(per) == 8 and tot is not None, path
    return per, tot


def table4(split):
    data = {m: [ego_log(EGO_LOG[split].format(run=f"{m}_seed{s}"), split) for s in SEEDS] for m in EGO_MODELS}
    keys = list(data[EGO_MODELS[0]][0][0])
    assert all(list(p) == keys for m in EGO_MODELS for p, _ in data[m]), "scene lists differ"
    head = ["scene"] + list(EGO_MODELS)
    rows_txt, rows_tex, rows_csv = [], [], []
    for tw, s in keys:
        v = {m: st.mean(p[(tw, s)] for p, _ in data[m]) for m in EGO_MODELS}
        best = min(v.values())
        rows_txt.append([f"{tw} #{s}"] + [fmt(v[m]) + ("*" if v[m] == best else "") for m in EGO_MODELS])
        rows_tex.append(" & ".join([f"{tw} \\#{s}"] + [bold(fmt(v[m]), v[m] == best) for m in EGO_MODELS])
                        + " \\\\")
        rows_csv.append([split, tw, s] + [round(v[m], 4) for m in EGO_MODELS])
    alls = {m: [t for _, t in data[m]] for m in EGO_MODELS}
    best = min(st.mean(a) for a in alls.values())
    rows_txt.append(["ALL"] + [pm(alls[m])[1] + ("*" if st.mean(alls[m]) == best else "") for m in EGO_MODELS])
    rows_tex.append(" & ".join(["ALL"] + [bold(pm(alls[m])[0], st.mean(alls[m]) == best) for m in EGO_MODELS])
                    + " \\\\")
    rows_csv.append([split, "ALL", "ALL"] + [round(st.mean(alls[m]), 4) for m in EGO_MODELS]
                    + [round(st.stdev(alls[m]), 4) for m in EGO_MODELS])
    header_tex = ["scene & " + " & ".join(m.replace("_", "\\_") for m in EGO_MODELS) + " \\\\"]
    write(f"ego_per_scene_{split}", header_tex, "lrrrr", rows_tex, rows_csv,
          ["split", "town", "scene"] + [f"{m}_minade6" for m in EGO_MODELS] + [f"{m}_std" for m in EGO_MODELS],
          rules_after=(len(keys) - 1,))
    show(f"4 ego task minADE6 per scene, {split} (m, seed mean of 3; * = best; ALL mean +/- std)", head, rows_txt)


# ---------------------------------------------------------------- 5
def path_len(p):
    """[..., T, 2] own-frame path from the origin -> length [...]."""
    q = np.concatenate([np.zeros(p.shape[:-2] + (1, 2)), p], axis=-2)
    return np.linalg.norm(np.diff(q, axis=-2), axis=-1).sum(-1)


def table5(cache, G):
    V = G["V2X"]
    ref = V[0]
    o = [np.argsort(t["tid"]) for t in V]
    tid = ref["tid"][o[0]]
    vals = {k: np.stack([t[k][oi] for t, oi in zip(V, o)]) for k in ("ade1", "cv_ade")}   # [seeds, N]
    scene, mb = ref["scene"][o[0]], (ref["vis"][o[0]] == 1) & ref["moving"][o[0]]
    groups = {"Town04 #30": mb & (scene == 30), "other 7 tuning scenes": mb & (scene != 30)}
    tw = cache["two_wheeler"][tid]
    speed = cache["speed"][tid] * FRAME_RATE_HZ
    head = ["quantity"] + list(groups)
    rows = []

    def row(name, f):
        rows.append([name] + [f(m) for m in groups.values()])

    row("moving (b) targets", lambda m: str(int(m.sum())))
    row("distinct vehicles", lambda m: str(len(set(cache["inst"][tid[m]].tolist()))))
    row("two-wheelers (% of targets)", lambda m: f"{100 * tw[m].mean():.1f}")
    row("distinct two-wheelers", lambda m: str(len(set(cache["inst"][tid[m & tw]].tolist()))))
    for q, name in ((5, "p5"), (25, "p25"), (50, "median"), (75, "p75"), (95, "p95"), (100, "max")):
        row(f"speed {name} (m/s)", lambda m, q=q: f"{np.percentile(speed[m], q):.2f}")
    for tname, tm in (("all", None), ("car", ~tw), ("two-wheeler", tw)):
        sel = (lambda m: m) if tm is None else (lambda m, tm=tm: m & tm)
        row(f"n {tname}", lambda m, sel=sel: str(int(sel(m).sum())))
        row(f"ADE1 E_V2X, {tname} (m)",
            lambda m, sel=sel: pm([float(v[sel(m)].mean()) for v in vals["ade1"]])[1] if sel(m).any() else "n/a")
        row(f"ADE CV, {tname} (m)", lambda m, sel=sel: fmt(float(vals["cv_ade"][0][sel(m)].mean()))
            if sel(m).any() else "n/a")
        row(f"E_V2X ADE1 < CV ADE, {tname} (% of targets, seed mean)",
            lambda m, sel=sel: f"{100 * np.mean([(a[sel(m)] < c[sel(m)]).mean() for a, c in zip(vals['ade1'], vals['cv_ade'])]):.1f}"
            if sel(m).any() else "n/a")
    assert np.allclose(vals["cv_ade"], vals["cv_ade"][0]), "CV differs between seeds"
    # overshoot: the only stored top-1 paths
    z = np.load("viz_env/pred/E_V2X_seed0_scene30_ego3.npz")
    meta = json.loads(str(z["meta"]))
    assert (meta["run"], meta["scene"], meta["split"], meta["ckpt"]) == ("E_V2X_seed0", 30, "tuning",
                                                                       "checkpoint_epoch12.pt"), meta
    p_tid, trajs, logits = z["tid"], z["trajs"], z["logits"]
    sel = (cache["vis"][p_tid] == 1) & (cache["speed"][p_tid] > 0)
    p_tid, top = p_tid[sel], trajs[sel][np.arange(sel.sum()), np.argmax(logits[sel], axis=1)]
    pos = {t: i for i, t in enumerate(V[0]["tid"].tolist())}
    assert all(t in pos for t in p_tid.tolist()) and (V[0]["scene"][[pos[t] for t in p_tid.tolist()]] == 30).all()
    fut, mask = cache["fut_own"][p_tid].astype(np.float64), cache["fut_mask"][p_tid]
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    assert (mask.sum(1) == last + 1).all(), "masked steps are not a prefix"
    keep = np.arange(mask.shape[1])[None] <= last[:, None]
    d = np.array([path_len(top[i][keep[i]]) - path_len(fut[i][keep[i]]) for i in range(len(p_tid))])
    two = cache["two_wheeler"][p_tid]
    sub = []
    for tname, m in (("all", np.ones(len(d), bool)), ("car", ~two), ("two-wheeler", two)):
        if m.any():
            sub.append([f"top-1 length - GT length, {tname} (n={int(m.sum())}): overshoot > {SHOOT_M:g} m / "
                        f"within / undershoot < -{SHOOT_M:g} m (%)",
                        f"{100 * (d[m] > SHOOT_M).mean():.1f} / {100 * (np.abs(d[m]) <= SHOOT_M).mean():.1f} / "
                        f"{100 * (d[m] < -SHOOT_M).mean():.1f}", "n/a (not stored)"])
    sub.append(["top-1 length - GT length, all: median (p25, p75) (m)",
                f"{np.median(d):+.2f} ({np.percentile(d, 25):+.2f}, {np.percentile(d, 75):+.2f})",
                "n/a (not stored)"])
    note = (f"overshoot rows: seed 0, ego CAV 3's samples only ({meta['n_samples']} samples, {len(d)} of the "
            f"{int(groups['Town04 #30'].sum())} moving (b) targets; viz_env/pred/E_V2X_seed0_scene30_ego3.npz)")
    rows += sub
    rows_tex = [" & ".join(r[0].replace("_", "\\_").replace("%", "\\%").replace("#", "\\#").replace(">", "$>$")
                           .replace("<", "$<$")
                           .replace("+/-", "$\\pm$") if i == 0 else c.replace("+/-", "$\\pm$").replace("%", "\\%")
                           for i, c in enumerate(r)) + " \\\\" for r in rows]
    header_tex = ["quantity (tuning, E\\_V2X seed 0--2, moving (b) targets) & Town04 \\#30 & other 7 tuning scenes \\\\"]
    write("town04_30_failure", header_tex, "lrr", rows_tex, [["town04_30_failure"] + r for r in rows],
          ["table", "quantity", "town04_30", "other_7_tuning_scenes"], rules_after=(3, 9, len(rows) - len(sub) - 1))
    with open(os.path.join(OUT, "town04_30_failure.tex"), "a") as f:
        f.write(f"% {note}\n")
    show("5 Town04 #30 failure facts (tuning; E_V2X seeds 0-2 vs constant velocity; " + note + ")", head, rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = load_env_cache()
    z = np.load(os.path.join(ENV_CACHE_DIR, "env_targets.npz"))
    cache["inst"], cache["two_wheeler"] = z["inst"], z["two_wheeler"]     # not in load_env_cache's keys
    splits = {}
    for split in ("tuning", "test"):
        G, metas, scenes = env_runs(split)
        for s in SEEDS:   # E_V2XRSU metas for the RSU-only cross-check of table 3
            p = f"runs/E_V2XRSU_seed{s}/eval_env_{split}_checkpoint_epoch12.npz"
            if os.path.exists(p):
                metas[f"E_V2XRSU_seed{s}"] = load(p)[1]
        splits[split] = (G, metas, scenes)
        table1(split, G, scenes)
        table2(split, G)
    table3(cache, splits)
    for split in ("tuning", "test"):
        table4(split)
    table5(cache, splits["tuning"][0])
    print(f"\nwrote {sorted(os.listdir(OUT))} to {OUT}/")


if __name__ == "__main__":
    main()
