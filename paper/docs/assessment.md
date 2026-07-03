# Assessment

## Goal: Measuring Value, Not Just Validity

The purpose of the derived event logs is not to exist, but to let a process analyst answer questions that no enterprise system can answer about manual factory work: where does worker time actually go, how fragmented is value-adding work, which improvement opportunities are supported by evidence, and how do workers performing the same task differ? The assessment is therefore organized around *delivered analytical value*. We first establish, briefly, that the logs are reliable enough to analyze (validity preconditions), and then spend the bulk of the assessment executing the analyses an operations analyst would actually run, reporting what they reveal, what they would trigger in practice, and where they hit the limits of the data. All analyses operate solely on the released artifacts and are reproducible from the published assessment code; no access to the source videos or any model API is required.

Three questions structure the assessment:

- **Q1 (Preconditions).** Are the logs trustworthy enough to base operational conclusions on?
- **Q2 (Analysis capability).** Which classes of process mining analysis do the logs support in practice, and what do they reveal in the six factories?
- **Q3 (Optimization evidence).** Do the logs produce concrete, prioritizable optimization findings, and how do these findings relate to the qualitative process reports generated independently by the pipeline?

**Setup.** 6 factories, 59 workers, 322 per-video event logs with 10,698 events, 13,304 transcript entries spanning 218.4 hours of footage. Computations use pandas and pm4py; the video-as-case notion is used for control-flow analyses; time-composition analyses use event-covered time. The process-category mapping (below) and all scripts are released with the artifacts.

## Q1: Validity Preconditions (Condensed)

A value assessment is meaningless on unreliable data, so we summarize the validity checks compactly; full details are in the released assessment materials. *Structure and vocabulary:* all 10,698 events are schema-complete, chronologically ordered, and conform 100% to the factory vocabularies (zero out-of-vocabulary labels; 89 to 97% of catalogue activities actually used). *Temporal fidelity:* median event coverage of the observed footage is about 95% for the 314 recordings up to one hour; coverage collapses (median 1.2%) on the seven multi-hour recordings, where the extraction stage truncates, so these recordings are excluded from all time statistics below. Event boundaries coincide with transcribed behavioral change points in 82% of cases (within 5 s). *Semantic fidelity:* a stratified manual audit (42 events, 42 transcript entries, seed 42) yields strict event precision 0.86 with zero contradicted labels and operational recall about 0.89; misses concentrate in the truncation regime. In short: within the dominant length regime, the logs are a faithful, conservatively extracted account of observed work, and the conclusions below inherit at most the uncertainty quantified here.

## Q2: Analysis Capability

We execute four analysis classes that together cover the main promises of process mining for manual work: performance analysis (A1), operational diagnostics (A2), control-flow analysis (A3), and resource comparison (A4). For A1 and A2, each of the 112 factory process labels is mapped to one of seven operational categories: value-adding transformation (VA), material handling and staging (MH), transport (TR), quality inspection (QI), cleaning and maintenance (CM), rework (RW), and documentation (DOC). The mapping is released and deliberately coarse; its role is the same as an activity-based-costing scheme in industrial engineering.

### A1: Work-Time Composition (Where Does Worker Time Go?)

Share of event-covered time per category and factory:

| Factory | VA | MH | TR | QI | CM | RW | DOC | Non-VA total |
|---|---|---|---|---|---|---|---|---|
| 001 | 93.1% | 4.1% | 1.8% | 0.0% | 0.2% | 0.0% | 0.9% | 6.9% |
| 002 | 96.3% | 0.1% | 2.3% | 0.2% | 0.0% | 0.0% | 1.1% | 3.7% |
| 003 | 82.1% | 0.4% | 0.0% | 0.0% | 6.3% | **10.8%** | 0.4% | 17.9% |
| 004 | 89.9% | 3.0% | 3.5% | 2.0% | 0.9% | 0.0% | 0.7% | 10.1% |
| 005 | 80.6% | 5.4% | 9.7% | 4.3% | 0.0% | 0.0% | 0.0% | 19.4% |
| 006 | 74.2% | 12.1% | 7.8% | 0.0% | 5.7% | 0.0% | 0.3% | **25.8%** |

This single table already delivers the core promise of the dataset: a quantified, comparable account of manual work that exists in no information system. Three findings stand out. First, the non-value-adding share varies by a factor of seven across factories (3.7% to 25.8%), immediately ranking where operational improvement effort should go: the foundry (006) loses one quarter of observed worker time to handling, transport, and cleaning, whereas the garment packaging factory (002) is already highly compact. Second, factory 003 spends **10.8% of observed work time on rework** (the process *Defect patching*, in which workers fill and patch surface defects on castings): a finding with direct economic meaning, pointing upstream to casting quality rather than to the patching stations themselves. Third, the machining factory (005) shows the highest transport share (9.7%) plus 4.3% inspection: machinists act as their own logistics and quality staff.

### A2: Fragmentation of Value-Adding Work (Operational Diagnostics)

Time shares alone understate the damage of interruptions: thirty 15-second replenishment breaks are operationally worse than one 7.5-minute one. Per worker, we collapse consecutive same-category events into runs and measure how often value-adding work is interrupted and how long uninterrupted VA runs last (medians per factory, workers with at least 30 minutes observed):

| Factory | Median VA share | Interruptions of VA per hour | Median uninterrupted VA run |
|---|---|---|---|
| 001 | 94.5% | 9.2 | 4.6 min |
| 002 | 97.3% | 3.9 | 8.8 min |
| 003 | 86.4% | 3.6 | 9.0 min |
| 004 | 91.3% | 10.6 | 3.6 min |
| 005 | 85.9% | 5.6 | 6.2 min |
| 006 | 87.0% | **14.4** | **3.1 min** |

Combining A1 and A2 separates two distinct problem profiles that would demand different interventions. Factory 001 loses little total time (6.9% non-VA) but is interrupted 9.2 times per hour by mostly short episodes (449 MH/TR episodes, mean 15 s): the problem is *material presentation at the workstation* (parts within reach, gravity bins), not travel distance. Factory 006 combines high frequency (14.4 interruptions/hour, 28.4 MH/TR episodes per observed hour) with high total loss (19.8% of time in MH/TR): here the problem is *layout and logistics*, and value-adding work never runs longer than about three minutes uninterrupted. This distinction (presentation problem versus logistics problem) is exactly the kind of diagnosis a lean consultant produces from days of stopwatch observation; the logs produce it from data that was recorded anyway.

### A3: Flow Visibility (Control-Flow Analysis)

Control-flow discovery on the raw activity level yields high-fitness but low-precision models (Inductive Miner, noise 0.2; token-based replay fitness 0.95 to 0.99, precision 0.06 to 0.27): traces are nearly all unique variants because workers interleave replenishment, transport, and cleaning with value-adding cycles at self-chosen points. This is a property of flexibly ordered manual work, not noise, and it delimits what discovery can deliver here: activity-level logs support *local* pattern analysis (directly-follows loops identify the repetitive work cycles; 4 to 17% of directly-follows pairs are self-loops) and drill-down, not readable end-to-end models. Collapsing to the process level of the two-level vocabulary changes the picture: with 12 to 24 phase labels and 2 to 26 segments per trace, discovered phase models reach precision up to 0.59 (factory 001, from 0.069 at activity level) at fitness above 0.96, and are small enough to read. The practical recipe for analysts: discover at phase level, quantify at activity level.

### A4: Worker Benchmarking (Resource Comparison)

Because the vocabulary is shared within a factory, workers performing the same dominant process are directly comparable, which turns the logs into an internal benchmarking instrument:

| Factory | Shared process | Workers | VA share range | Mean VA run range |
|---|---|---|---|---|
| 001 | Manual mechanical assembly | 6 | 89.7 to 97.8% | 1.7 to 6.7 min |
| 001 | Metal stamping | 3 | 90.6 to 98.5% | 3.2 to 7.9 min |
| 003 | Surface finishing and polishing | 5 | **67.3 to 96.9%** | 3.3 to 13.7 min |
| 004 | Garment ironing | 2 | **52.8 to 77.1%** | 3.5 to 4.1 min |
| 004 | Overlock seaming | 4 | 87.8 to 96.5% | 1.5 to 10.3 min |
| 005 | Manual lathe machining | 2 | 95.0 to 95.8% | 6.9 to 9.4 min |

The spreads are the finding. Five workers doing surface finishing in factory 003 range from 67% to 97% VA share; the two ironing workers in factory 004 differ by 24 points. Such gaps localize either genuine best practices worth standardizing (workstation setups that keep parts in reach) or hidden structural differences between nominally identical stations, and they define exactly where a follow-up observation is worth its cost. Conversely, the two lathe machinists in 005 are within one point of each other: their station design, not their behavior, drives the factory's 19.4% non-VA share.

Additionally, a Pareto view within VA time sharpens improvement targeting: the top three value-adding activities absorb 27% to 62% of VA time per factory (62% in factory 003: scraping, filing, and paste application), so tooling or automation investments can be evaluated against precisely quantified time bases.

## Q3: Optimization Evidence and Triangulation

The pipeline independently produces qualitative optimization observations (S3 factory reports, generated from transcripts without any event log). Confronting those claims with the log-derived numbers tests whether the logs add value beyond the reports, in three directions:

| Report claim (S3, qualitative) | Log evidence (quantitative) | Relation |
|---|---|---|
| 001: "most pervasive bottleneck is the lack of dedicated material handlers; operators constantly ... fetch raw materials, transport finished bags" | 449 MH/TR episodes, 14.6/h, but mean only 15 s and 5.9% of time | **Refined**: frequency, not time share, is the problem; points to material presentation, not headcount |
| 006: "excessive manual material handling acts as a facility-wide bottleneck" | 19.8% of observed time in MH/TR, 28.4 episodes/h, VA runs of 3.1 min | **Confirmed and quantified**: the largest measured capacity loss in the corpus |
| 003: "workers frequently halt value-adding operations to carry heavy metal trays, push wheelbarrows" | MH/TR is only 0.4% of time; the 003 process vocabulary contains **no transport label** | **Exposed a blind spot**: transport exists in the transcripts but cannot be represented in this factory's vocabulary; the log under-measures it |
| 003 report does not mention rework as a theme | *Defect patching* = 10.8% of observed time | **New finding**: the single largest improvement lever in 003 appears only in the log |

The three relations are exactly the value proposition of event logs over narrative reports: quantification enables prioritization (which of the six factories' many plausible observations deserves investment first), refinement changes the intervention (presentation aids versus dedicated handlers in 001), and the two 003 rows show both a genuine log-only discovery (rework share) and an honest failure mode (the controlled vocabulary is a measurement instrument: what it does not name, the log cannot see). The latter yields a concrete methodological rule for this class of pipelines: vocabulary induction should be required to cover a standard set of operational categories (transport, waiting, rework, inspection) in every factory, precisely so that their absence is a measured zero rather than a blind spot.

## What the Logs Cannot (Yet) Answer

Being explicit about non-capabilities is part of the value assessment. (i) *Throughput and yield*: events carry no product identifiers or counts, so pieces per hour and first-pass yield are out of reach without object-centric extensions. (ii) *Calendar-time utilization*: timestamps live on a synthetic per-worker axis; shift patterns, breaks between recordings, and true machine utilization are not represented. (iii) *Cross-worker flows*: handoffs are visible as events (e.g., *hand part to coworker for inspection*) but material is not tracked across workers, so factory-level flow analysis stops at the worker boundary. (iv) *Waiting*: the extraction prompt filters non-operational footage, which also removes most waiting; waiting-as-waste is therefore systematically under-measured. Each limitation maps to a concrete pipeline extension (object references, wall-clock anchoring, cross-camera correlation, an explicit waiting category), which we consider the main value of having run the assessment.

## Findings

- **F1 (Q1).** The logs pass validity preconditions in the dominant length regime (100% vocabulary conformance, about 95% median coverage, audited precision 0.86 with zero contradictions); the multi-hour truncation failure is isolated and excluded.
- **F2 (A1).** The logs quantify where manual work time goes, revealing a sevenfold spread in non-value-adding share across factories (3.7% to 25.8%) and a 10.8% rework share in factory 003 that no qualitative report surfaced.
- **F3 (A2).** Combining time shares with interruption frequency separates presentation problems (001: frequent 15-second interruptions, low total loss) from logistics problems (006: 14 interruptions/hour and one quarter of time lost), leading to different interventions.
- **F4 (A3, A4).** Discovery is useful at phase level (precision up to 0.59) while activity level supports drill-down; shared vocabularies enable worker benchmarking that localizes 24 to 30-point efficiency spreads on identical tasks.
- **F5 (Q3).** Triangulation against the pipeline's own qualitative reports shows the logs confirm, refine, and extend narrative findings, and it exposes the controlled vocabulary as a measurement instrument whose gaps (a missing transport label) become measurable blind spots.

## Threats to Validity

*Construct:* the category mapping (112 labels to 7 categories) embeds judgment calls (e.g., whether mold coating is value-adding); the mapping is released, and the headline findings (003 rework, 006 handling load, benchmarking spreads) are robust to the contestable assignments because they rest on unambiguous labels. Time composition measures *event-covered* time; the excluded waiting and the truncated recordings bias non-VA shares downward, so the reported losses are lower bounds. *Internal:* validity numbers rest on a single-author audit of limited size (42+42), and the transcript, not the video, is the reference for both fidelity and, transitively, all downstream analyses; a video-grounded audit remains necessary before operational decisions. *External:* six factories of one corpus, discrete manufacturing only; the analysis classes transfer, the numbers do not. *Conclusion:* benchmarking compares observed windows of different lengths per worker; workers below 30 minutes of footage were excluded, and spreads should be read as hypotheses for targeted follow-up, not verdicts on individuals; this is also an ethical requirement, since the same analyses could be misused for individual performance rating.

## Reproducibility

All tables derive from the released per-video logs, transcripts, and factory reports via the published scripts (`value_analysis.py` for A1, A2, A4 and the composition/fragmentation tables; `pm_utility.py` for A3; `assess.py` for Q1), with fixed seeds for all sampling. The category mapping is versioned alongside the code.
