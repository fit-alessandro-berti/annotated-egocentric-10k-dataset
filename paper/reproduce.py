#!/usr/bin/env python3
"""Recalculate descriptive results from frozen artifacts, using only Python's standard library.

No model APIs, discovery algorithms, timestamp repair, or new audit judgments are used.
The category map and run definitions are inherited from value_analysis.py.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import importlib.util
import platform
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = datetime(2000, 1, 1)
CATS = ('VA', 'MH', 'TR', 'QI', 'CM', 'RW', 'DOC')
LEGACY_RE = re.compile(r'\[(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]')
INTERVAL_RE = re.compile(r'\[(\d+:\d{2}(?::\d{2})?)\s*[-\u2013]\s*(\d+:\d{2}(?::\d{2})?)\]')


def seconds(t):
    return sum(int(x) * 60 ** i for i, x in enumerate(t.split(':')[::-1]))


def union_length(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return sum(b-a for a, b in merged)


def legacy_round(value):
    """Match NumPy scalar one-decimal rounding used in the original script."""
    return round(value * 10) / 10


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def tex(s):
    s = str(s).replace('\u2014', ', ').replace('\u2013', ' to ').replace('≥', 'at least ')
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}'}.get(c,c) for c in s)


def vocabulary(path):
    text = path.read_text()
    def extract(a, b):
        part = text.split(a, 1)[1].split(b, 1)[0]
        return [ln.strip()[2:].strip() for ln in part.splitlines() if ln.strip().startswith(('* ', '- '))]
    return extract('Factory Process Labels','Factory Activity Catalogue'), extract('Factory Activity Catalogue','Optimization-Relevant Process Observations')


def category_map(path):
    mapping = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'm':
            cat, labels = map(ast.literal_eval, node.value.args)
            for label in labels:
                if label in mapping and mapping[label] != cat:
                    raise ValueError(f'Conflicting category: {label}')
                mapping[label] = cat
    return mapping


def audit_rows(path):
    rows = []
    sample = 'event'
    for line in path.read_text().splitlines():
        if line.startswith('## Recall'):
            sample = 'entry'
        if not re.match(r'\| \d+ \|', line):
            continue
        parts = [s.strip() for s in line.strip('|').split('|')]
        n, source, interval = parts[:3]
        if sample == 'event':
            label, verdict, note = parts[3:6]
        else:
            verdict, note = parts[3:5]
            label = ''
        rows.append(dict(sample=sample, item=int(n), source=source, interval=interval, label=label, verdict=verdict, note=note))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'generated')
    args = p.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    material = root/'paper/docs/assessment_materials'
    mapping = category_map(material/'value_analysis.py')
    events, videos, intervals, maps, artifacts = [], [], [], [], []
    vocab = {}
    for report in sorted((root/'factory_process_mining_reports').glob('*.txt')):
        pv, av = vocabulary(report)
        vocab[report.stem] = (pv, av)
        maps += [dict(factory=report.stem, process=x, category=mapping.get(x, "unmapped")) for x in pv]
    for path in sorted((root/'process_mining_event_logs').glob('*/*/*.csv')):
        fac, worker, video = path.parts[-3], path.parts[-2], path.stem
        transcript = root/'raw_transcriptions'/fac/worker/(video+'.txt')
        text = transcript.read_text()
        parsed = INTERVAL_RE.findall(text)
        legacy = LEGACY_RE.findall(text)
        formats, transitions = [], []
        previous_format = None
        for i, (a,b) in enumerate(parsed, 1):
            form = ('HMS' if a.count(':') == 2 else 'MS') + '/' + ('HMS' if b.count(':') == 2 else 'MS')
            if previous_format and form != previous_format:
                transitions.append(f'{i}:{previous_format}>{form}')
            previous_format = form
            formats.append(form)
            intervals.append(dict(video=video, entry=i, raw_start=a, raw_end=b, format=form, literal_start_s=seconds(a), literal_end_s=seconds(b)))
        rows = read_csv(path)
        ev = []
        for i, row in enumerate(rows, 1):
            a = int((datetime.fromisoformat(row['start_timestamp'])-BASE).total_seconds())
            b = int((datetime.fromisoformat(row['end_timestamp'])-BASE).total_seconds())
            ev.append(dict(factory=fac, worker=worker, video=video, row=i, start=a, end=b, duration=b-a, activity=row['activity'], process=row['process'], category=mapping[row['process']]))
        events.extend(ev)
        eend = max((e['end'] for e in ev), default=0)
        tend = max((seconds(b) for a,b in parsed), default=0)
        lend = max((seconds(b) for a,b in legacy), default=0)
        bounds = {seconds(x) for ab in legacy for x in ab}
        matches = sum(any(abs(e[k]-t)<=5 for t in bounds) for e in ev for k in ('start','end'))
        flags = []
        if tend > 3600: flags.append('literal_transcript_endpoint_over_1h')
        if any(x != 'HMS/HMS' for x in formats): flags.append('contains_MM:SS')
        if eend > 3600: flags.append('event_endpoint_over_1h')
        videos.append(dict(factory=fac, worker=worker, video=video, events=len(ev), transcript_entries=len(parsed), legacy_entries=len(legacy), formats=';'.join(sorted(set(formats))), format_transitions=';'.join(transitions), last_transcript_endpoint=parsed[-1][1] if parsed else '', literal_transcript_end_s=tend, legacy_transcript_end_s=lend, event_end_s=eend, event_union_s=union_length((e['start'],e['end']) for e in ev), summed_event_s=sum(e['duration'] for e in ev), legacy_span_s=max(lend,eend), literal_span_s=max(tend,eend), boundary_matches=matches, boundary_total=2*len(ev), retained_time_analysis=eend<=3600, reason='retained by original event-end rule' if eend<=3600 else 'excluded: maximum event endpoint > 3600 s', flags=';'.join(flags), source_duration_s='unavailable'))
    write_csv(out/'events.csv', events)
    write_csv(out/'inclusion_manifest.csv', videos)
    write_csv(out/'transcript_intervals.csv', intervals)
    write_csv(out/'category_mapping_by_factory.csv', maps)
    retained = {v['video'] for v in videos if v['retained_time_analysis']}
    keep = [e for e in events if e['video'] in retained]
    # Reproduce the original category runs: start a new run at each clip, sum row
    # durations, ignore gaps, then concatenate run lists before counting transitions.
    workers = []
    groups = defaultdict(list)
    for e in keep: groups[e['factory'],e['worker']].append(e)
    for (fac,w), rows in sorted(groups.items()):
        rows.sort(key=lambda e:(e['video'],e['start']))
        total = sum(e['duration'] for e in rows)
        if total < 1800: continue
        runs = []
        for e in rows:
            if runs and runs[-1]['video']==e['video'] and runs[-1]['category']==e['category']:
                runs[-1]['duration'] += e['duration']
            else: runs.append(dict(video=e['video'],category=e['category'],duration=e['duration']))
        va = [r['duration'] for r in runs if r['category']=='VA']
        transitions = [(a,b) for a,b in zip(runs,runs[1:]) if a['category']=='VA' and b['category']!='VA']
        processes = Counter()
        for e in rows: processes[e['process']] += e['duration']
        dominant = sorted(processes, key=lambda x:(-processes[x], x))[0]
        workers.append(dict(factory=fac,worker=w,summed_event_s=total,dominant_process=dominant,va_share=legacy_round(100*sum(va)/((total/3600)*3600)),va_to_nonva=len(transitions),cross_clip_transitions=sum(a['video']!=b['video'] for a,b in transitions),transitions_per_event_hour=legacy_round(len(transitions)/(total/3600)),mean_va_run_min=legacy_round(st.mean(va)/60) if va else 0))
    write_csv(out/'worker_metrics.csv', workers)
    summaries, structures, benchmarks = [], [], []
    for fac, (pv,av) in vocab.items():
        all_e=[e for e in events if e['factory']==fac]
        vs=[v for v in videos if v['factory']==fac]
        es=[e for e in keep if e['factory']==fac]
        ws=[w for w in workers if w['factory']==fac]
        comp={c:sum(e['duration'] for e in es if e['category']==c) for c in CATS}
        total=sum(comp.values()); mhtr=[e for e in es if e['category'] in ('MH','TR')]
        procs=Counter()
        for e in es: procs[e['process']] += e['duration']
        traces=[]; pairs=loops=0
        for v in vs:
            trace=tuple(e['activity'] for e in all_e if e['video']==v['video'])
            traces.append(trace); pairs += max(0,len(trace)-1)
            loops += sum(a==b for a,b in zip(trace,trace[1:]))
        structures.append(dict(factory=fac,workers=len({e['worker'] for e in all_e}),videos=len(vs),events=len(all_e),activity_vocabulary=len(av),activities_used=len({e['activity'] for e in all_e}),process_vocabulary=len(pv),processes_used=len({e['process'] for e in all_e}),empty_logs=sum(not v['events'] for v in vs),missing_fields=sum(any(e[k]=='' for k in ('activity','process','factory','worker')) for e in all_e),out_of_vocabulary=sum(e['activity'] not in av or e['process'] not in pv for e in all_e),negative_durations=sum(e['duration']<0 for e in all_e),zero_durations=sum(e['duration']==0 for e in all_e),out_of_order=sum(a['start']>b['start'] for a,b in zip(all_e,all_e[1:]) if a['video']==b['video']),adjacent_overlaps=sum(b['start']<a['end'] for a,b in zip(all_e,all_e[1:]) if a['video']==b['video']),legacy_coverage_pct=100*sum(v['event_union_s'] for v in vs)/sum(v['legacy_span_s'] for v in vs),literal_coverage_pct=100*sum(v['event_union_s'] for v in vs)/sum(v['literal_span_s'] for v in vs),legacy_boundary_pct=100*sum(v['boundary_matches'] for v in vs)/sum(v['boundary_total'] for v in vs),unique_traces=len(set(traces)),directly_follows_pairs=pairs,self_loops=loops))
        summary=dict(factory=fac,retained_videos=len({e['video'] for e in es}),retained_events=len(es),summed_event_s=total,eligible_workers=len(ws))
        summary.update({c+'_pct':round(100*comp[c]/total,1) for c in CATS})
        summary.update(median_worker_va_pct=st.median(w['va_share'] for w in ws),median_transitions_per_event_hour=st.median(w['transitions_per_event_hour'] for w in ws),median_worker_mean_va_run_min=st.median(w['mean_va_run_min'] for w in ws),mhtr_rows=len(mhtr),mhtr_s=sum(e['duration'] for e in mhtr),mhtr_rows_per_event_hour=len(mhtr)*3600/total,mhtr_mean_row_s=st.mean(e['duration'] for e in mhtr),top_processes='; '.join(f'{name}: {100*d/total:.1f}%' for name,d in procs.most_common(3)))
        summaries.append(summary)
        for proc in sorted({w['dominant_process'] for w in ws}):
            group=[w for w in ws if w['dominant_process']==proc]
            if len(group)>=2:
                benchmarks.append(dict(factory=fac,dominant_process=proc,workers=len(group),min_va=min(w['va_share'] for w in group),max_va=max(w['va_share'] for w in group),min_mean_run=min(w['mean_va_run_min'] for w in group),max_mean_run=max(w['mean_va_run_min'] for w in group)))
    write_csv(out/'factory_metrics.csv',summaries)
    write_csv(out/'structural_metrics.csv',structures)
    write_csv(out/'worker_comparisons.csv',benchmarks)
    transport=[e for e in events if e['factory']=='factory_003' and e['activity'] in ('transport loaded tray to storage rack','transport bulk material via wheelbarrow')]
    write_csv(out/'factory003_transport.csv',transport)
    audit=audit_rows(material/'audit_verdicts.md')
    write_csv(out/'audit_judgments.csv',audit)
    # Archive inventory hashes allow independent verification that source outputs
    # were not repaired or regenerated as part of the revision.
    for directory in ('raw_transcriptions','worker_process_summaries','factory_process_mining_reports','process_mining_event_logs','merged_process_mining_event_logs'):
        for path in sorted((root/directory).rglob('*')):
            if path.is_file():
                data=path.read_bytes()
                artifacts.append(dict(layer=directory,path=str(path.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    audit_path=material/'audit_verdicts.md'
    audit_data=audit_path.read_bytes()
    artifacts.append(dict(layer='assessment_audit', path=str(audit_path.relative_to(root)), bytes=len(audit_data), sha256=hashlib.sha256(audit_data).hexdigest()))
    write_csv(out/'artifact_manifest.csv',artifacts)
    spec = importlib.util.spec_from_file_location('merge_stage', root/'05_merge_event_log_csvs.py')
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    merge_checks=[]
    for fac in vocab:
        combined=[]
        for worker_dir in sorted((root/'process_mining_event_logs'/fac).iterdir()):
            if not worker_dir.is_dir(): continue
            fields, reconstructed = merge.merge_worker_csvs(worker_dir)
            target=root/'merged_process_mining_event_logs'/fac/(worker_dir.name+'.csv')
            actual=read_csv(target)
            assert reconstructed == actual, f'Merged worker mismatch: {target}'
            combined.extend(reconstructed)
            merge_checks.append(dict(path=str(target.relative_to(root)),events=len(actual),matches_S5=True))
        target=root/'merged_process_mining_event_logs'/fac/(fac+'.csv')
        actual=read_csv(target)
        assert merge.build_factory_rows(combined) == actual, f'Merged factory mismatch: {target}'
        merge_checks.append(dict(path=str(target.relative_to(root)),events=len(actual),matches_S5=True))
    write_csv(out/'merge_verification.csv',merge_checks)
    # Use unrounded duration shares for the standalone vector figure.
    colors={'VA':'blue!45','MH':'orange!70','TR':'green!55!black','QI':'violet!65',
            'CM':'gray!60','RW':'red!65','DOC':'yellow!70!black'}
    chart=[r'\begin{tikzpicture}[x=0.085cm,y=0.62cm,font=\scriptsize]']
    for i, fac in enumerate(vocab):
        es=[e for e in keep if e['factory']==fac];total=sum(e['duration'] for e in es)
        x=0
        chart.append(r'\node[left] at (0,'+str(-i)+r'){\textbf{'+fac[-3:]+'}};')
        for c in CATS:
            width=100*sum(e['duration'] for e in es if e['category']==c)/total
            chart.append(r'\fill['+colors[c]+'] ('+str(x)+','+str(-i-0.28)+') rectangle ('+str(x+width)+','+str(-i+0.28)+');')
            x+=width
    for x in (0,25,50,75,100):
        chart.append(r'\node[below] at ('+str(x)+',-5.6){'+str(x)+r'\%};')
    for i,c in enumerate(CATS):
        x=i*14
        chart.append(r'\fill['+colors[c]+'] ('+str(x)+',-7) rectangle ('+str(x+3)+',-6.6);')
        chart.append(r'\node[right] at ('+str(x+3)+',-6.8){'+c+'};')
    chart.append(r'\end{tikzpicture}')
    (out/'composition_figure.tex').write_text('\n'.join(chart)+'\n')
    discovery_path=out/'activity_discovery.json'
    if discovery_path.exists():
        discovery=json.loads(discovery_path.read_text())
        lines=[r'\begin{table*}[!t]\begin{adjustwidth}{-\extralength}{0cm}\centering\caption{Recalculation with the original activity-level discovery settings. Fitness is average trace fitness from token replay; precision is token-based precision.}\label{tab:discovery}',
               r'\small\begin{tabular}{@{}lrrr@{}}\toprule Factory & Cases & Fitness & Precision \\ \midrule']
        for fac,row in discovery['factories'].items():
            lines.append(r'\textbf{'+fac[-3:]+f"}} & {row['cases']} & {row['fitness']:.4f} & {row['precision']:.4f} "+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{adjustwidth}\end{table*}', r'\FloatBarrier', 'PM4Py version for this recalculation: '+tex(discovery['pm4py_version'])+'.']
        (out/'discovery_table.tex').write_text('\n'.join(lines)+'\n')

    prompts=['annotation_prompt.txt','worker_process_summary_prompt.txt','factory_process_mining_prompt.txt','annotation_to_event_log_prompt.txt']
    config_files=prompts+[x.name for x in sorted(root.glob('0[1-5]_*.py'))]
    config_hashes={x:hashlib.sha256((root/x).read_bytes()).hexdigest() for x in config_files}
    summary=dict(python=platform.python_version(),source_artifact_counts=dict(Counter(a['layer'] for a in artifacts)),source_code_sha256=config_hashes,events=len(events),retained_events=len(keep),retained_videos=len(retained),legacy_transcript_entries=sum(v['legacy_entries'] for v in videos),format_aware_transcript_entries=len(intervals),legacy_span_s=sum(v['legacy_span_s'] for v in videos),literal_span_s=sum(v['literal_span_s'] for v in videos),union_event_s=sum(v['event_union_s'] for v in videos),summed_event_s=sum(e['duration'] for e in events),retained_event_s=sum(e['duration'] for e in keep),format_flagged_videos=sum('contains_MM:SS' in v['flags'] for v in videos),over_1h_transcript_videos=sum(v['literal_transcript_end_s']>3600 for v in videos),excluded_videos=[v['video'] for v in videos if not v['retained_time_analysis']],process_vocabulary_entries=len(maps),distinct_process_strings=len({m['process'] for m in maps}),transport_003_events=len(transport),transport_003_seconds=sum(e['duration'] for e in transport),cross_clip_transitions=sum(w['cross_clip_transitions'] for w in workers),audit_counts={sample:dict(Counter(a['verdict'] for a in audit if a['sample']==sample)) for sample in ('event','entry')})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Typeset appendices are derived mechanically from the archived judgments and
    # mapping. The source audit's unsupported truncation wording is marked below.
    appendix=[]
    for sample,title in [('event','Archived event judgments'),('entry','Archived transcript-entry judgments')]:
        appendix.append(r'\subsection{'+title+'}')
        rows=[a for a in audit if a['sample']==sample]
        for part,start in enumerate(range(0,len(rows),21),1):
            appendix += [r'\begin{table*}[!t]\begin{adjustwidth}{-\extralength}{0cm}',
                         r'\caption{'+title+f' (part {part} of 2).'+r'}\label{tab:audit_'+sample+f'_{part}'+'}',
                         r'\footnotesize\setlength{\tabcolsep}{3pt}',
                         r'\begin{tabularx}{\linewidth}{@{}r p{0.16\linewidth} p{0.20\linewidth} L@{}}',
                         r'\toprule No. & Source & Interval; verdict & Label and archived note \\ \midrule']
            for a in rows[start:start+21]:
                note=a['note']
                if a['sample']=='entry' and a['item']==10:
                    note=note.replace('(multi-hour recording; truncation regime)','(archive interprets this as truncation; that cause is unverified)')
                content=a['label']+('; ' if a['label'] and note else '')+note
                source=re.sub(r'^(\d{3})',lambda m:r'\textbf{'+m[1]+'}',tex(a['source']))
                appendix.append(f"{a['item']} & {source} & {tex(a['interval'])}; {a['verdict']} & {tex(content)} "+r'\\')
            appendix += [r'\bottomrule\end{tabularx}\end{adjustwidth}\end{table*}']
        appendix += [r'\FloatBarrier', r'Tables~\ref{tab:audit_'+sample+r'_1} and~\ref{tab:audit_'+sample+r'_2} preserve the '+('event' if sample=='event' else 'transcript-entry')+' judgments.']
    (out/'audit_appendix.tex').write_text('\n'.join(appendix)+'\n')
    lines=[]
    for part,start in enumerate(range(0,len(maps),57),1):
        lines += [r'\begin{table*}[!t]\begin{adjustwidth}{-\extralength}{0cm}',
                  r'\caption{Operational-category map'+f' (part {part} of 2).'+r'}\label{tab:mapping_'+str(part)+'}',
                  r'\small\renewcommand{\arraystretch}{0.96}\begin{tabularx}{\linewidth}{@{}l L l@{}}',r'\toprule Factory & Process label & Category \\ \midrule']
        for m in maps[start:start+57]:
            lines.append(r'\textbf{'+m['factory'][-3:]+f"}} & {tex(m['process'])} & {m['category']} "+r'\\')
        lines += [r'\bottomrule\end{tabularx}\end{adjustwidth}\end{table*}']
    lines += [r'\FloatBarrier', r'Tables~\ref{tab:mapping_1} and~\ref{tab:mapping_2} reproduce all 113 factory-specific process entries.']
    (out/'mapping_appendix.tex').write_text('\n'.join(lines)+'\n')
    for name in prompts:
        text=(root/name).read_text().replace('\u2014',', ').replace('\u2013',' to ').replace('→','->').replace('≥','>=')
        (out/name).write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_code_sha256'},indent=2))

if __name__=='__main__': main()
