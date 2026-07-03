#!/usr/bin/env python3
"""Value-oriented assessment: what can an analyst learn from the video-derived logs?"""
import pandas as pd, numpy as np, json, re

SCRATCH = "/tmp/claude-1000/-home-javert-annotated-egocentric-10k-dataset/57dc18f3-0252-4f13-887f-83e45f675ce9/scratchpad"
df = pd.read_parquet(f"{SCRATCH}/events.parquet")
df["dur"] = df["end"] - df["start"]

# exclude the 7 truncated multi-hour recordings from time statistics
span = df.groupby("video")["end"].max()
long_videos = set(span[span > 3600].index)
df = df[~df["video"].isin(long_videos)].copy()
print(f"analysed videos: {df['video'].nunique()} (excluded {len(long_videos)} >1h)")

# ---- category mapping of the 112 process labels ----
CAT = {}
def m(cat, labels):
    for l in labels: CAT[l] = cat
# factory_001
m("VA", ["Metal stamping","Manual mechanical assembly","Component bagging","Semi-automated press assembly",
         "Metal forming","Component kitting","Structural frame assembly"])
m("MH", ["Material replenishment","Scrap disposal"])
m("TR", ["Finished goods transport"])
m("DOC",["Production tracking"])
m("CM", ["Workstation cleaning","Machine maintenance"])
# factory_002
m("VA", ["Fabric folding","Garment bagging","Garment ironing","Sub-assembly marking","Garment tagging",
         "Fabric pairing","Hanger assembly","Cardboard wrapping","Fabric edge alignment","Pattern matching",
         "Box packing","Box assembly","Fabric bundling","Box sealing","Package labeling","Master bagging",
         "Cardboard sleeve packaging","Fabric pinning","Thread snipping"])
m("MH", ["Fabric untangling"])
m("TR", ["Material transport","Cart loading"])
m("QI", ["Garment quality inspection"])
m("DOC",["Floor auditing"])
# factory_003
m("VA", ["Surface finishing and polishing","Wax pattern assembly","Metal casting separation",
         "Injection molding","Component sub-assembly","Machine pressing","Spot painting","Thermal welding"])
m("RW", ["Defect patching"])
m("CM", ["Mold preparation and cleaning"])
m("MH", ["Material sorting and sifting"])
m("DOC",["Inventory verification"])
# factory_004
m("VA", ["Overlock seaming","Garment folding","Thread trimming","Fabric spreading","Edge binding attachment",
         "Garment hemming","Continuous fabric chaining","Bulk fabric cutting","Piping attachment",
         "Curvilinear seam joining","Heat press labeling","Tape feeding","Fabric measuring","Bulk box assembly"])
m("MH", ["Pick-and-pack sorting","Scrap material management"])
m("QI", ["Quality inspection"])
m("DOC",["Production data logging"])
# factory_005
m("VA", ["manual lathe machining","drill press machining","manual hardware pre-assembly",
         "motor labeling and nameplate attachment","batch pneumatic fastening","stator coil binding",
         "stator lead wire finishing and soldering","component deburring and polishing","milling machine operation",
         "finished goods packaging","automated stator winding","CNC machine tending",
         "semi-automatic stator wire insertion","motor packaging and strapping"])
m("MH", ["part sorting and staging"])
m("TR", ["material transport and logistics"])
m("QI", ["dimensional quality inspection"])
# factory_006
m("VA", ["Sand mold creation","Compression molding","Mold assembly and closing","Core and insert placement",
         "Machine sanding","Metal casting separation","Wooden component scraping","Raw material mixing",
         "Part mechanical modification","Dough kneading","Punch press operation","Wooden component assembly",
         "Chemical dough mixing","Powder sifting","Machine extrusion","Post-processing edge finishing",
         "Pattern cleaning and coating"])
m("MH", ["Component staging and arrangement","Floor-based part sorting","Mold track advancement"])
m("TR", ["Bulk material transport"])
m("CM", ["Machine cleaning and maintenance","Workspace clearing"])
m("DOC",["Manual production logging"])

df["cat"] = df["process"].map(CAT)
unmapped = df[df["cat"].isna()]["process"].unique()
assert len(unmapped) == 0, unmapped

# ---- 1. time composition per factory ----
print("\n=== TIME COMPOSITION (share of covered event time, %) ===")
comp = df.pivot_table(index="factory", columns="cat", values="dur", aggfunc="sum").fillna(0)
comp_pct = (100*comp.div(comp.sum(axis=1), axis=0)).round(1)
print(comp_pct.to_string())
print("non-VA share:", (100-comp_pct["VA"]).round(1).to_dict())

# ---- 2. interruption / fragmentation of value-adding work ----
print("\n=== FRAGMENTATION OF VA WORK ===")
rows=[]
for (fac,w), g in df.groupby(["factory","worker"]):
    g = g.sort_values(["video","start"])
    hours = g["dur"].sum()/3600
    if hours < 0.5: continue
    # collapse consecutive same-category events per video into runs
    runs=[]
    for _, vg in g.groupby("video"):
        cur=None
        for _,r in vg.iterrows():
            if cur and r["cat"]==cur[0]:
                cur[1]+=r["dur"]
            else:
                if cur: runs.append(cur)
                cur=[r["cat"], r["dur"]]
        if cur: runs.append(cur)
    va_runs=[d for c,d in runs if c=="VA"]
    inter = [c for i,(c,d) in enumerate(runs) if c!="VA" and 0<i<len(runs)-0 and runs[i-1][0]=="VA"]
    rows.append(dict(factory=fac, worker=w, hours=round(hours,2),
                     va_share=round(100*sum(va_runs)/ (hours*3600),1),
                     interruptions_per_h=round(len(inter)/hours,1),
                     mean_va_run_min=round(np.mean(va_runs)/60,1) if va_runs else 0,
                     top_interrupter=pd.Series(inter).mode()[0] if inter else "-"))
frag = pd.DataFrame(rows)
print(frag.groupby("factory").agg(workers=("worker","count"),
      med_va_share=("va_share","median"),
      med_intr_h=("interruptions_per_h","median"),
      med_run_min=("mean_va_run_min","median")).round(1).to_string())

# ---- 3. transport & replenishment cadence (batching evidence) ----
print("\n=== TRANSPORT/REPLENISHMENT EPISODES ===")
for fac, g in df.groupby("factory"):
    tr = g[g["cat"].isin(["TR","MH"])]
    if not len(tr): continue
    n = len(tr); mean_d = tr["dur"].mean()
    obs_h = g["dur"].sum()/3600
    print(f"{fac}: {n} MH/TR episodes, mean {mean_d:.0f}s, {n/obs_h:.1f} per observed hour, "
          f"total {tr['dur'].sum()/3600:.2f}h ({100*tr['dur'].sum()/g['dur'].sum():.1f}%)")

# ---- 4. worker benchmarking on shared processes ----
print("\n=== WORKER BENCHMARKING (same dominant process) ===")
for fac, g in df.groupby("factory"):
    dom = g.groupby("worker").apply(lambda x: x.groupby("process")["dur"].sum().idxmax(), include_groups=False)
    for proc, ws in dom.groupby(dom).groups.items():
        if len(ws) < 2: continue
        sel = frag[(frag["factory"]==fac) & (frag["worker"].isin(ws))]
        if len(sel) < 2: continue
        print(f"{fac} | {proc}: {len(sel)} workers, VA share {sel['va_share'].min()}-{sel['va_share'].max()}%, "
              f"mean VA run {sel['mean_va_run_min'].min()}-{sel['mean_va_run_min'].max()} min")

# ---- 5. within-VA activity Pareto (improvement targeting) ----
print("\n=== TOP-3 VA ACTIVITIES SHARE ===")
for fac, g in df.groupby("factory"):
    va = g[g["cat"]=="VA"]
    top3 = va.groupby("activity")["dur"].sum().nlargest(3)
    print(f"{fac}: {100*top3.sum()/va['dur'].sum():.0f}% of VA time in top-3 activities: {list(top3.index)}")

frag.to_csv(f"{SCRATCH}/fragmentation.csv", index=False)
comp_pct.to_csv(f"{SCRATCH}/composition.csv")
