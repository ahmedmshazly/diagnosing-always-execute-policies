"""Rebuild the SN Computer Science analysis from saved evaluation artifacts.

No policy is trained here. Workload-level intervals are conditional on the
saved policies. Training seeds are displayed individually, not pooled as
independent workload observations.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ENV_ORDER = ["default", "env_tight_fixedrisk", "env_heavytail_tight",
             "env_cascade", "env_cascade_broken"]
ENV_NAMES = ["Default", "Tight capacity", "High-value mixture*",
             "Cascade: corrected", "Cascade: original"]
BOOT_SEED = 20260418


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def holm(pvalues):
    """Holm step-down adjusted p-values, retaining p=1 hypotheses."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite p-values in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def family(rows):
    """Thirty learned comparisons and four distinct hand references."""
    result = []
    for row in rows:
        if row["algo"] == "baseline":
            continue
        if row["env"] == "env_cascade_broken" and row["algo"] == "hand":
            continue  # same reference and same dynamics as corrected cascade
        p = float(row["p"])
        if not np.isfinite(p):
            # Historical files contain NaN for identically zero differences.
            # Only this explicitly identified case is converted to p=1.
            if float(row["delta_vs_ae"]) != 0 or float(row["exec_frac"]) != 1:
                raise ValueError(f"Unexplained missing p-value: {row}")
            p = 1.0
        result.append({**row, "p": p})
    if len(result) != 34:
        raise ValueError(f"Expected 34 comparisons, got {len(result)}")
    for row, adj in zip(result, holm([r["p"] for r in result])):
        row["p_holm_34"] = float(adj)
    return result


def bootstrap_ci(x, statistic=np.mean, reps=10000):
    x = np.asarray(x)
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, len(x), size=(reps, len(x)))
    values = np.array([statistic(x[i]) for i in indices])
    return np.quantile(values, [.025, .975]).tolist()


def probe_report(path):
    rows = read_csv(path)
    seeds = sorted({int(r["seed"]) for r in rows})
    # Enumerate all ordered bootstrap samples: 5^5 = 3125.
    indices = np.array(list(itertools.product(range(len(seeds)), repeat=len(seeds))))
    result = []
    for key in rows[0]:
        if not key.startswith("adv_"):
            continue
        groups = [np.array([float(r[key]) for r in rows if int(r["seed"]) == s])
                  for s in seeds]
        if len({len(g) for g in groups}) != 1:
            raise ValueError("Equal-sized episode clusters expected")
        means = np.array([g.mean() for g in groups])
        lo, hi = np.quantile(means[indices].mean(axis=1), [.025, .975])
        result.append({"action": key[4:], "mean": float(means.mean()),
                       "cluster_ci_low": float(lo), "cluster_ci_high": float(hi),
                       "source_episodes": len(seeds), "sampled_states": len(rows),
                       "futures_per_state": 16})
    return result


def paired_report(path):
    rows = read_csv(path)
    a = np.array([float(r["rl_util"]) for r in rows])
    b = np.array([float(r["reflex_util"]) for r in rows])
    # The saved difference is rounded independently of each saved utility.
    d = np.array([float(r["diff"]) for r in rows])
    assert np.allclose(d, a-b, atol=1.01e-4, rtol=0)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    ci90 = [float(d.mean() - stats.t.ppf(.95, n-1)*se),
            float(d.mean() + stats.t.ppf(.95, n-1)*se)]
    score = (d > 0) + .5*(d == 0)
    dz = abs(d.mean()/d.std(ddof=1))
    def power(nn):
        critical = stats.t.ppf(.975, nn-1)
        return float(stats.nct.sf(critical, nn-1, dz*np.sqrt(nn)) +
                     stats.nct.cdf(-critical, nn-1, dz*np.sqrt(nn)))
    return dict(n=n, mean_rl=float(a.mean()), mean_reflex=float(b.mean()),
                difference=float(d.mean()), difference_ci95=bootstrap_ci(d),
                wilcoxon_p=float(stats.wilcoxon(d).pvalue),
                t_p=float(stats.ttest_1samp(d, 0).pvalue),
                iqm_rl=float(stats.trim_mean(a, .25)),
                iqm_reflex=float(stats.trim_mean(b, .25)),
                iqm_rl_ci95=bootstrap_ci(a, lambda x: stats.trim_mean(x, .25)),
                iqm_reflex_ci95=bootstrap_ci(b, lambda x: stats.trim_mean(x, .25)),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()),
                strict_win_fraction=float((d > 0).mean()),
                paired_win_score=float(score.mean()), paired_win_score_ci95=bootstrap_ci(score),
                mean_difference_ci90=ci90, exploratory_tost_3=ci90[0]>-3 and ci90[1]<3,
                exploratory_tost_5=ci90[0]>-5 and ci90[1]<5,
                observed_dz=dz, paired_t_sensitivity_n50=power(50),
                paired_t_sensitivity_n250=power(250)), a, b


def training_inventory():
    from scripts.hardening.eval_taxonomy import MATRIX
    output = []
    for config, _, patterns in MATRIX:
        for algo, pats in patterns.items():
            paths = sorted({p for pat in pats.split() for p in glob.glob(pat) if Path(p).is_dir()})
            for run in paths:
                p = Path(run)
                seed = int(p.name.split("seed")[-1])
                curve = p / "learning_curve.csv"
                manifest = p / "run_manifest.json"
                complete = curve.exists() and manifest.exists() and (p / "policy_final.pt").exists()
                record = dict(env=Path(config).stem, algo=algo, init_seed=seed,
                              run_dir=run, checkpoint="policy_best_by_val.pt",
                              checkpoint_sha256=hashlib.sha256((p/"policy_best_by_val.pt").read_bytes()).hexdigest(),
                              completion_record=bool(complete), updates="", env_steps="",
                              validation_n="", validation_seeds="", schedule="", note="")
                if complete:
                    cr = read_csv(curve)
                    mf = json.loads(manifest.read_text())
                    record.update(updates=len(cr), env_steps=int(cr[-1]["env_steps_cumulative"]),
                                  validation_n=len(mf["seed_list"]),
                                  validation_seeds=f'{min(mf["seed_list"])}:{max(mf["seed_list"])+1}')
                    record["schedule"] = "fixed 100 jobs / 300 steps" if algo=="PPO" else "; ".join(
                        f'{r["num_jobs"]} jobs / exponential scale {r["max_steps"]}'
                        for i,r in enumerate(cr) if i==0 or r["stage"] != cr[i-1]["stage"])
                else:
                    log = Path("results/hardening/batch_logs") / (p.name + ".log")
                    log_text = log.read_text()
                    checkpoints = re.findall(r"\[val @ update\s+(\d+)\]", log_text)
                    steps = re.findall(r"steps=(\d+)", log_text)
                    record.update(updates=f'>={max(map(int, checkpoints))}', env_steps=f'>={steps[-1]}',
                                  validation_n=20, validation_seeds="250:270 (driver/log)",
                                  schedule="planned 150 updates; 100 jobs / exponential scale 300",
                                  note="Incomplete record: log reaches validation update 125; no final curve or manifest. Saved best checkpoint first selected at update 25.")
                output.append(record)
    assert len(output) == 30
    return output


def figures(rows50, rows250, a, b, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":9, "font.family":"DejaVu Sans", "pdf.fonttype":42,
                         "ps.fonttype":42, "axes.spines.top":False, "axes.spines.right":False})
    colors = ["#12558a", "#a34f00"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios":[1.8, 1]})
    thresholds = np.unique(np.r_[a,b])
    ax[0].step(thresholds, [(a>=t).mean() for t in thresholds], where="pre", color=colors[0], label="REINFORCE (seed 7)")
    ax[0].step(thresholds, [(b>=t).mean() for t in thresholds], where="pre", color=colors[1], linestyle="--", label="Reflex")
    ax[0].set(xlabel="Episode utility threshold", ylabel="Fraction at or above threshold", ylim=(0,1.02))
    ax[0].legend(frameon=False, fontsize=8)
    d=a-b
    bars=ax[1].bar(["Win", "Tie", "Loss"],[(d>0).sum(),(d==0).sum(),(d<0).sum()],
                   color=[colors[0],"#999999",colors[1]], edgecolor="black", linewidth=.4)
    for bar in bars:
        ax[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(int(bar.get_height())),ha="center",fontsize=9)
    ax[1].set(ylabel="Paired workloads", ylim=(0,120))
    for i, axis in enumerate(ax):
        axis.text(0,1.04,f"({chr(97+i)})",transform=axis.transAxes,fontweight="bold")
    fig.tight_layout(w_pad=2)
    for ext in ["pdf", "eps"]:
        fig.savefig(directory/f"Fig1.{ext}",bbox_inches="tight")
    plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(6.6,3.2),sharey=True)
    for axis, rows, n in zip(axes,[rows50,rows250],[50,250]):
        axis.axvline(0,color="#555555",linewidth=.8)
        for yi, env in enumerate(ENV_ORDER):
            for ai, algo in enumerate(["REINFORCE","PPO"]):
                rr=[next(r for r in rows if r["env"]==env and r["policy"]==f"{algo}:seed{s}") for s in [7,11,13]]
                yy=yi+(-.13 if ai==0 else .13)
                incomplete=env=="env_heavytail_tight" and ai==0
                axis.scatter([float(r["delta_vs_ae"]) for r in rr], [yy-.045,yy,yy+.045],
                             marker="o" if ai==0 else "^",s=26,
                             facecolors="white" if incomplete else colors[ai],
                             edgecolors=colors[ai],linewidths=.9,
                             label=algo if yi==0 else None, zorder=3)
        axis.set(xlabel="Utility gain over always-execute",xlim=(-9,137),xticks=[0,40,80,120])
        axis.grid(axis="y",color="#dddddd",linewidth=.4)
        axis.text(0,1.04,f"({'a' if n==50 else 'b'})  {n} workloads",transform=axis.transAxes)
    axes[0].set_yticks(range(5), ENV_NAMES)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False,fontsize=8,loc="upper right")
    fig.tight_layout(w_pad=1.8)
    for ext in ["pdf","eps"]:
        fig.savefig(directory/f"Fig2.{ext}",bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("results/sncs"))
    ap.add_argument("--figures",type=Path,default=Path("figures"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    datasets=[]
    inputs={}
    for n, historical in [(50,"taxonomy.csv"),(250,"taxonomy_n250.csv")]:
        new=args.out/f"rechecked_n{n}.csv"
        path=new if new.exists() else Path("results/hardening")/historical
        rows=read_csv(path)
        if len(rows)!=40:
            raise ValueError(f"Expected 40 rows in {path}")
        corrected=family(rows)
        write_csv(args.out/f"holm_n{n}.csv",corrected)
        inputs[f"n{n}"]=str(path)
        datasets.append(rows)
        if new.exists():
            old=read_csv(Path("results/hardening")/historical)
            indexed={(r["env"],r["policy"]):r for r in old}
            differences=[]
            for r in rows:
                before=indexed[(r["env"],r["policy"])]
                for field in ["util","delta_vs_ae","fail","compl","exec_frac"]:
                    if abs(float(r[field])-float(before[field]))>1e-6:
                        differences.append({"env":r["env"],"policy":r["policy"],"field":field,"old":before[field],"new":r[field]})
            inputs[f"n{n}_reevaluation_differences"]=differences
    paired_path=args.out/"power_rliable_perseed.csv"
    if not paired_path.exists():
        paired_path=Path("results/hardening/power_rliable_perseed.csv")
    paired,a,b=paired_report(paired_path)
    probe=probe_report("results/hardening/local_optimality.csv")
    write_csv(args.out/"probe_cluster_intervals.csv",probe)
    write_csv(args.out/"training_inventory.csv",training_inventory())
    report={"inputs":inputs,"paired_comparison":paired,"probe":probe,
            "inference":"Workload-level uncertainty conditional on saved policies; not training-seed inference."}
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    figures(*datasets,a,b,args.figures)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
