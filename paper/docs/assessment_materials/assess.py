#!/usr/bin/env python3
"""Assessment of the video-derived event logs (structural, vocabulary, temporal, PM utility)."""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/javert/annotated-egocentric-10k-dataset")
LOGS = ROOT / "process_mining_event_logs"
TRANS = ROOT / "raw_transcriptions"
REPORTS = ROOT / "factory_process_mining_reports"

TS_RE = re.compile(r"\[(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]")

def hms_to_s(t):
    h, m, s = t.split(":")
    return int(h)*3600 + int(m)*60 + int(s)

def parse_report(path):
    """Extract process labels and activity catalogue bullets from a factory report."""
    text = path.read_text(encoding="utf-8")
    def section(name, nxt):
        m = re.search(re.escape(name) + r"\s*\n(.*?)(?:\n" + re.escape(nxt) + r"\s*\n)", text, re.S)
        return m.group(1) if m else ""
    proc_sec = section("Factory Process Labels", "Factory Activity Catalogue")
    act_sec = section("Factory Activity Catalogue", "Optimization-Relevant Process Observations")
    bullets = lambda s: [re.sub(r"^[\*\-•]\s*", "", ln).strip() for ln in s.splitlines() if ln.strip().startswith(("*", "-", "•"))]
    return set(bullets(proc_sec)), set(bullets(act_sec))

def parse_ts(x):
    # format: 2000-01-01 HH:MM:SS
    return hms_to_s(x.split(" ")[1])

results = {}
per_video_rows = []   # for pm4py
audit_pool = []       # (factory, worker, video, event dict, transcript text)

factories = sorted(d.name for d in LOGS.iterdir() if d.is_dir())
for fac in factories:
    proc_vocab, act_vocab = parse_report(REPORTS / f"{fac}.txt")
    st = dict(videos=0, events=0, empty_logs=0,
              bad_order=0, neg_dur=0, zero_dur=0, overlap=0,
              act_in_vocab=0, proc_in_vocab=0,
              covered_s=0.0, span_s=0.0, gap_s=0.0,
              boundary_match=0, boundary_total=0,
              acts_used=Counter(), procs_used=Counter(),
              trans_entries=0, workers=set(), durations=[])
    st["proc_vocab_size"] = len(proc_vocab)
    st["act_vocab_size"] = len(act_vocab)
    for worker_dir in sorted((LOGS / fac).iterdir()):
        if not worker_dir.is_dir(): continue
        worker = worker_dir.name
        st["workers"].add(worker)
        for csv_path in sorted(worker_dir.glob("*.csv")):
            video = csv_path.stem
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            st["videos"] += 1
            if not rows:
                st["empty_logs"] += 1
                continue
            # transcript
            tpath = TRANS / fac / worker / f"{video}.txt"
            ttext = tpath.read_text(encoding="utf-8") if tpath.exists() else ""
            spans = [(hms_to_s(a), hms_to_s(b)) for a, b in TS_RE.findall(ttext)]
            st["trans_entries"] += len(spans)
            trans_end = max((b for _, b in spans), default=0)
            boundary_set = set()
            for a, b in spans:
                boundary_set.add(a); boundary_set.add(b)

            prev_end = None
            prev_start = None
            events = []
            for r in rows:
                s0, e0 = parse_ts(r["start_timestamp"]), parse_ts(r["end_timestamp"])
                dur = e0 - s0
                events.append((s0, e0, r["activity"], r["process"]))
                st["events"] += 1
                st["durations"].append(dur)
                if dur < 0: st["neg_dur"] += 1
                elif dur == 0: st["zero_dur"] += 1
                if prev_start is not None and s0 < prev_start:
                    st["bad_order"] += 1
                if prev_end is not None and s0 < prev_end:
                    st["overlap"] += 1
                prev_start, prev_end = s0, e0
                if r["activity"] in act_vocab: st["act_in_vocab"] += 1
                if r["process"] in proc_vocab: st["proc_in_vocab"] += 1
                st["acts_used"][r["activity"]] += 1
                st["procs_used"][r["process"]] += 1
                # boundary alignment within +-5 s
                tol = 5
                for bnd, key in ((s0, "s"), (e0, "e")):
                    st["boundary_total"] += 1
                    if any(abs(bnd - t) <= tol for t in boundary_set):
                        st["boundary_match"] += 1
                per_video_rows.append(dict(factory=fac, worker=worker, video=video,
                                           start=s0, end=e0, activity=r["activity"], process=r["process"]))
            # temporal coverage vs transcript span
            span = max(trans_end, max(e for _, e, _, _ in events))
            covered = sorted((s, e) for s, e, _, _ in events if e > s)
            merged = []
            for s0, e0 in covered:
                if merged and s0 <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e0)
                else:
                    merged.append([s0, e0])
            cov = sum(e0 - s0 for s0, e0 in merged)
            st["covered_s"] += cov
            st["span_s"] += span
            st["gap_s"] += max(0, span - cov)
            audit_pool.append((fac, worker, video, rows, ttext))
    st["workers"] = len(st["workers"])
    results[fac] = st

# aggregate + print
def pct(a, b): return 100.0 * a / b if b else 0.0

summary = {}
tot = Counter()
import statistics
for fac, st in results.items():
    durs = st["durations"]
    row = dict(
        workers=st["workers"], videos=st["videos"], events=st["events"],
        empty_logs=st["empty_logs"],
        trans_entries=st["trans_entries"],
        proc_vocab=st["proc_vocab_size"], act_vocab=st["act_vocab_size"],
        acts_used=len(st["acts_used"]), procs_used=len(st["procs_used"]),
        act_vocab_util=pct(len([a for a in st["acts_used"] if a in set()]) if False else len(set(st["acts_used"]) & set()), 1),
        act_conf=pct(st["act_in_vocab"], st["events"]),
        proc_conf=pct(st["proc_in_vocab"], st["events"]),
        neg_dur=st["neg_dur"], zero_dur=pct(st["zero_dur"], st["events"]),
        bad_order=st["bad_order"], overlap_pct=pct(st["overlap"], st["events"]),
        coverage=pct(st["covered_s"], st["span_s"]),
        boundary_align=pct(st["boundary_match"], st["boundary_total"]),
        med_dur=statistics.median(durs) if durs else 0,
        mean_dur=statistics.mean(durs) if durs else 0,
        span_h=st["span_s"]/3600.0,
        covered_h=st["covered_s"]/3600.0,
        top10_share=pct(sum(c for _, c in st["acts_used"].most_common(10)), st["events"]),
    )
    summary[fac] = row
    for k in ("workers","videos","events","trans_entries","empty_logs","neg_dur","bad_order"):
        tot[k] += st[k]
    tot["span_s"] += st["span_s"]; tot["covered_s"] += st["covered_s"]
    tot["act_in"] += st["act_in_vocab"]; tot["proc_in"] += st["proc_in_vocab"]
    tot["bmatch"] += st["boundary_match"]; tot["btot"] += st["boundary_total"]
    tot["zero"] += st["zero_dur"]; tot["overlap"] += st["overlap"]

print(json.dumps(summary, indent=1))
print("TOTALS", dict(tot))
print("overall act conf %.2f  proc conf %.2f  coverage %.2f  boundary %.2f  zero %.2f  overlap %.2f" % (
    pct(tot["act_in"], tot["events"]), pct(tot["proc_in"], tot["events"]),
    pct(tot["covered_s"], tot["span_s"]), pct(tot["bmatch"], tot["btot"]),
    pct(tot["zero"], tot["events"]), pct(tot["overlap"], tot["events"])))

# vocabulary utilisation
for fac, st in results.items():
    proc_vocab, act_vocab = parse_report(REPORTS / f"{fac}.txt")
    used = set(st["acts_used"])
    print(fac, "act catalogue", len(act_vocab), "used-in-log", len(used & act_vocab),
          "util %.1f%%" % pct(len(used & act_vocab), len(act_vocab)),
          "| proc vocab", len(proc_vocab), "used", len(set(st["procs_used"]) & proc_vocab))

# dump flat events for pm4py stage
import pandas as pd
df = pd.DataFrame(per_video_rows)
df.to_parquet("/tmp/claude-1000/-home-javert-annotated-egocentric-10k-dataset/57dc18f3-0252-4f13-887f-83e45f675ce9/scratchpad/events.parquet")
print("flat events:", len(df))
