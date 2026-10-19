#!/usr/bin/env python3
"""Process-mining utility assessment with pm4py (case notion = video)."""
import warnings, json, argparse
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd
import pm4py

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "generated")
args = parser.parse_args()
SCRATCH = args.output_dir
df = pd.read_parquet(f"{SCRATCH}/events.parquet")
df["time:timestamp"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["end"], unit="s")
df["start_timestamp"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["start"], unit="s")
df = df.rename(columns={"video": "case:concept:name", "activity": "concept:name"})

out = {}
for fac, g in df.groupby("factory"):
    g = g.sort_values(["case:concept:name", "start", "end"]).copy()
    log = pm4py.format_dataframe(g, case_id="case:concept:name", activity_key="concept:name",
                                 timestamp_key="time:timestamp")
    n_cases = g["case:concept:name"].nunique()
    n_acts = g["concept:name"].nunique()
    variants = pm4py.get_variants(log)
    n_var = len(variants)
    # trace length stats
    tl = g.groupby("case:concept:name").size()
    # DFG density
    dfg, sa, ea = pm4py.discover_dfg(log)
    dfg_edges = len(dfg)
    # inductive miner with noise
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
    fit = pm4py.fitness_token_based_replay(log, net, im, fm)
    prec = pm4py.precision_token_based_replay(log, net, im, fm)
    out[fac] = dict(cases=n_cases, activities=n_acts, variants=n_var,
                    var_ratio=round(n_var / n_cases, 3),
                    trace_len_med=float(tl.median()), trace_len_mean=round(float(tl.mean()), 1),
                    dfg_edges=dfg_edges,
                    dfg_density=round(dfg_edges / (n_acts * n_acts), 4),
                    fitness=round(fit["average_trace_fitness"], 4),
                    precision=round(prec, 4))
    print(fac, out[fac], flush=True)

print(json.dumps(out, indent=1))
(SCRATCH / "activity_discovery.json").write_text(json.dumps({"provenance": "Recalculated from frozen logs with the original activity-level configuration; historical output and dependency versions unavailable.", "noise_threshold": 0.2, "pm4py_version": pm4py.__version__, "factories": out}, indent=2) + "\n")

# also: worker-level case notion, directly-follows repetition structure
print("\n--- self-loop share of DF pairs (video cases) ---")
for fac, g in df.groupby("factory"):
    g = g.sort_values(["case:concept:name", "start"])
    pairs = 0; loops = 0
    for _, cg in g.groupby("case:concept:name"):
        acts = cg["concept:name"].tolist()
        for a, b in zip(acts, acts[1:]):
            pairs += 1
            if a == b: loops += 1
    print(fac, f"df_pairs={pairs} self_loops={loops} ({100*loops/pairs:.1f}%)")
