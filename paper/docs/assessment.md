# Assessment

## Goals and Evaluation Framework

The pipeline produces event logs by *inference* rather than by extraction from a system of record. An assessment must therefore answer a question that classical log extraction can take for granted: *to what extent can the generated events be trusted, and for which analyses are they fit?* Because no independent ground-truth annotation exists for the source videos (producing one at corpus scale is exactly the cost the pipeline is designed to avoid) we assess the corpus through a framework of five complementary levels, ordered from purely syntactic to fully semantic and pragmatic. Each level has explicit criteria, an operationalization, and a quantitative result; together they address RQ1 (levels L1-L2), RQ2 (levels L3-L4), and RQ3 (level L5).

| Level | Criterion | Question | Reference |
|---|---|---|---|
| L1 | Structural validity | Are the logs well-formed event data? | log only |
| L2 | Vocabulary conformance & utilization | Is the closed-world labeling contract satisfied, and is the vocabulary adequate? | log + vocabularies |
| L3 | Temporal fidelity | Do events cover the observed footage, and do their boundaries align with observed change points? | log + transcript |
| L4 | Semantic fidelity | Do labels correctly describe the referenced behavior (precision), and is operational behavior captured (recall)? | manual audit against transcript |
| L5 | Process mining utility | Do standard techniques yield meaningful results, and at which abstraction level? | discovery experiments |

The chain of evidence deserves emphasis: L3 and L4 validate the *transcript→log* step against the transcripts, which are themselves model-generated. Validating the *video→transcript* step requires watching footage and is addressed only indirectly here (via the transcription prompt's uncertainty-marking requirement); a video-grounded audit is discussed under threats to validity.

**Setup.** The assessed corpus comprises 6 factories, 59 workers, 322 per-video event logs (all non-empty) with 10,698 events, and the corresponding 322 transcriptions with 13,304 timestamped entries (321 transcripts contain parseable timestamped entries) spanning 218.4 hours of transcript-covered footage. All computations were performed with Python 3.13, pandas 3.0.4, and pm4py 2.7.23; audit samples were drawn with a fixed random seed (42), stratified per factory. The video-as-case notion is used throughout unless stated otherwise.

## L1: Structural Validity

**Criteria.** Every event must parse against the declared schema; timestamps must be well-formed with `start ≤ end`; events within one video log must be chronologically ordered; events of one sequential worker stream should not overlap (a wearer performs one activity at a time, so overlaps indicate boundary estimation errors).

**Results.** All 10,698 events parse and are schema-complete (no missing values). There are **zero** negative-duration events, **zero** zero-duration events, and **zero** ordering violations (event starts are monotonically non-decreasing within every video log). Pairwise overlaps between consecutive events occur in **35 of 10,698 events (0.33%)**, concentrated in factories 001 (0.87%) and 002 (0.97%) and absent in the remaining four factories. No per-video log is empty, i.e., the extraction stage never degenerated to the admissible-but-vacuous output `{"events":[]}`.

**Finding F1.** The generation contract of stage S4 (strict JSON schema plus post-hoc validation) is sufficient to obtain structurally sound event data at scale; residual defects are rare, local boundary artifacts rather than systematic malformations.

## L2: Vocabulary Conformance and Utilization

**Criteria.** (i) *Conformance*: every `activity` (resp. `process`) value must be exactly one label of the factory's activity catalogue (resp. process label list), the closed-world contract of S4. (ii) *Utilization*: the vocabularies should be neither too narrow (which would force distortive label coercion) nor grossly over-generated (unused labels indicate hallucinated catalogue entries or over-splitting).

**Results.**

| Factory | Events | Activity vocab. | Activities used | Utilization | Process vocab. | Processes used | Label conformance |
|---|---|---|---|---|---|---|---|
| 001 | 2,288 | 131 | 125 | 95.4% | 13 | 13 | 100% / 100% |
| 002 | 1,541 | 75 | 67 | 89.3% | 24 | 24 | 100% / 100% |
| 003 | 2,449 | 55 | 52 | 94.5% | 12 | 12 | 100% / 100% |
| 004 | 1,893 | 71 | 65 | 91.5% | 22 | 22 | 100% / 100% |
| 005 | 1,103 | 102 | 97 | 95.1% | 18 | 17 | 100% / 100% |
| 006 | 1,424 | 61 | 59 | 96.7% | 24 | 24 | 100% / 100% |

Conformance is **100% for both attributes in all six factories**: not a single generated event carries an out-of-vocabulary label. Utilization is high (89-97% of catalogue activities appear in the logs; all process labels are used except one in factory 005), indicating that S3 induced vocabularies that are both grounded and close to minimal. Label usage is right-skewed, as expected for repetitive manual work: the ten most frequent activities account for 50-67% of events per factory.

**Finding F2.** The controlled-vocabulary mechanism works as designed: it fully eliminates label hallucination at extraction time (RQ1) while retaining fine-grained, almost fully utilized label sets. This contrasts with free-labeling LLM extraction, where label drift across documents is a known failure mode, and it is the property that makes the logs aggregatable across videos and workers at all.

## L3: Temporal Fidelity

**Criteria.** (i) *Coverage*: the fraction of the observed time span (per video: from 0 to the maximum of the last transcript timestamp and the last event end) that is covered by the union of event intervals. Coverage need not be 100%, because the extraction prompt deliberately excludes non-operational footage (walking without transport, phone use, camera artifacts), but it should be high for work-dense recordings. (ii) *Boundary alignment*: the fraction of event start/end points that coincide (±5 s) with a transcript entry boundary, i.e., with a change point detected in S1. Low alignment would mean S4 invents boundaries unanchored in observed behavior.

**Results.**

| Factory | Observed span (h) | Covered (h) | Coverage | Boundary alignment | Median event dur. (s) |
|---|---|---|---|---|---|
| 001 | 52.8 | 50.1 | 94.8% | 93.1% | 20 |
| 002 | 57.4 | 16.2 | 28.2% | 71.8% | 17 |
| 003 | 23.3 | 21.7 | 93.0% | 86.8% | 13 |
| 004 | 16.8 | 15.5 | 92.1% | 90.4% | 14 |
| 005 | 20.2 | 5.1 | 25.0% | 75.0% | 8 |
| 006 | 48.0 | 6.8 | 14.1% | 62.1% | 12 |
| **All** | **218.4** | **115.2** | **52.8%** | **82.1%** | **14** |

The aggregate coverage of 52.8% initially appears alarming, but stratifying by recording length localizes the deficit precisely:

| Video length | n | Median coverage | Median events per transcript entry |
|---|---|---|---|
| ≤ 20 min | 243 | **95.0%** | 0.80 |
| 20-60 min | 71 | **95.4%** | 0.72 |
| 1-2 h | 1 | 18.4% | 0.72 |
| > 2 h | 6 | **1.2%** | 0.56 |

Coverage is uniformly high (median ≈ 95%) for the 314 videos up to one hour and collapses for the seven multi-hour recordings (Pearson correlation between span and coverage ratio: r = −0.62). Manual inspection of the worst case (a 20-hour transcript in factory 002 with 0.19 h of events) shows the transcript's gap regions contain both genuinely non-operational stretches *and* clearly operational work (e.g., multi-hour garment-packaging episodes) for which no events were emitted: on very long transcripts, the extraction stage emits events for only an initial portion and effectively truncates, consistent with generation-length limits rather than with deliberate filtering. The three factories with low aggregate coverage (002, 005, 006) are exactly those whose observed hours are dominated by a few such long recordings.

Boundary alignment is 82.1% overall (93.1% in the best factory), confirming that S4 predominantly anchors events at S1-detected change points as instructed; the remaining boundaries stem from the sanctioned splitting of composite transcript entries.

**Finding F3.** Temporal fidelity is high (near-complete coverage with change-point-anchored boundaries) within the length regime that dominates the corpus (≤ 1 h; 98% of videos), and the failure mode outside that regime is sharply characterized: *long-input truncation of the extraction stage*. The practical mitigations are transcript chunking or windowed extraction for long recordings; until then, coverage should be reported per log, and the seven affected recordings excluded from performance analyses.

## L4: Semantic Fidelity (Manual Audit)

**Protocol.** Since transcripts verbalize the videos, the correctness of the transcript→event mapping can be audited by humans without re-watching footage. We drew two stratified random samples (seed 42, 7 items per factory each): (A) a **precision sample** of 42 events, each judged against all transcript entries overlapping the event interval, is the (activity, process) pair a correct description of the transcribed behavior in that interval?; (B) a **recall sample** of 42 transcript entries, each first classified as *operational* or *non-operational* (walking without transport, phone use, camera artifacts, idle observation), then checked for overlap with at least one event. Judgments were made by one author; the sample identifiers and verdicts are released with the artifacts.

**Results, precision.** Of 42 audited events, **36 (85.7%) are fully correct**: the activity label is a faithful, appropriately granular description of the transcribed behavior (e.g., event *level powder surface with wooden scraper* over the entry "picks up a flat wooden tool and aggressively scrapes it across…"). The remaining **6 (14.3%) are partially supported**: the label is plausible and consistent with the surrounding work context but not verifiable from the overlapping entry alone, typically because the transcript entry says only "the repetitive assembly process continues" and the label inherits its specificity from earlier context, or because a composite transcript entry was split and the label describes one constituent of the interval. **No audited event contradicts its transcript evidence (0/42).** Strict precision is thus 0.86 (95% Wilson interval: 0.72-0.93), and no outright labeling errors were observed.

**Results, recall.** Of 42 audited transcript entries, 34 are covered by at least one event and 8 are uncovered. Of the uncovered entries, 4 are non-operational and thus *correctly* excluded (a short walking sequence; a phone check; camera shake while walking; brief idle wandering), while 4 describe operational or borderline-operational behavior that was missed: aligning fabric stacks (inside a multi-hour recording, i.e., the L3 truncation regime), staging tied bundles onto a tray, operating a machine control panel, and inspecting a part. Estimated recall on operational entries is therefore **34/38 ≈ 0.89** (Wilson interval: 0.76-0.96), and half of the identified misses are attributable to the long-video truncation already quantified at L3 rather than to per-entry extraction failures.

**Finding F4.** The transcript→event step is semantically conservative: it essentially never assigns a wrong label (no contradictions in the audit), errs toward context-dependent rather than false labels, correctly filters non-operational footage, and misses operational content mainly through the long-input mechanism of F3. For a fully automatic pipeline, precision ≈ 0.86 (strict) / 1.0 (no-contradiction) and operational recall ≈ 0.89 substantially exceed what the authors expected from free-form LLM extraction, and we attribute this primarily to the closed vocabulary and the transcript-anchoring constraints.

## L5: Process Mining Utility

**Protocol.** For each factory we build an event log with the video-as-case notion and apply standard techniques in pm4py: variant analysis, directly-follows graph (DFG) construction, and process discovery with the Inductive Miner infrequent variant (noise threshold 0.2), evaluated by token-based replay fitness and precision. Discovery is run at two abstraction levels made possible by the two-level vocabulary: the fine-grained **activity level**, and the **process level**, obtained by collapsing consecutive events with identical process labels into segments.

**Results, activity level.**

| Factory | Cases | Acts. | Variants | Med. trace len. | DFG edges | Fitness | Precision |
|---|---|---|---|---|---|---|---|
| 001 | 97 | 125 | 97 | 19 | 594 | 0.980 | 0.069 |
| 002 | 54 | 67 | 53 | 20 | 239 | 0.991 | 0.060 |
| 003 | 72 | 52 | 70 | 21.5 | 305 | 0.987 | 0.091 |
| 004 | 51 | 65 | 51 | 27 | 336 | 0.982 | 0.067 |
| 005 | 22 | 97 | 22 | 41 | 241 | 0.954 | 0.268 |
| 006 | 26 | 59 | 26 | 45 | 272 | 0.954 | 0.080 |

Every trace is (nearly) a unique variant (variant/case ratio 0.97-1.0) and DFGs are sparse (2.6-11.3% of possible edges), the signature of flexibly ordered, interleaved manual work rather than of noise: workers weave material replenishment, transport, and cleaning into value-adding cycles at self-chosen points. Consequently, discovered models are high-fitness but low-precision, the classical over-generalization regime for logs of this kind. Directly-follows self-loops account for 4.3-16.8% of DF pairs, reflecting genuinely repetitive work cycles that survive the anti-repetition aggregation of S1. The activity-level logs are thus best suited to *frequency and performance analysis* (activity time budgets, waiting/transport shares, workstation comparisons) and *local* control-flow patterns, rather than to end-to-end model discovery.

**Results, process level.** Collapsing to process segments yields short traces (median 2-26 segments) over 12-24 labels:

| Factory | Process labels | Med. segments/case | Fitness | Precision |
|---|---|---|---|---|
| 001 | 13 | 6 | 0.966 | **0.591** |
| 002 | 24 | 7 | 0.968 | 0.149 |
| 003 | 12 | 2 | 0.997 | 0.264 |
| 004 | 22 | 11 | 0.990 | 0.093 |
| 005 | 17 | 4 | 0.992 | 0.320 |
| 006 | 24 | 26 | 0.996 | 0.137 |

Precision improves by a factor of 1.5-8.6 (e.g., 0.069 → 0.591 in factory 001) at essentially unchanged fitness, confirming that the two-level vocabulary provides a usable abstraction ladder: phase-level models are discoverable and interpretable, while activity-level logs support drill-down analysis beneath them. Factories whose process vocabularies are broad relative to their trace lengths (002, 004, 006) remain low-precision even at this level, suggesting their process label sets over-partition the work, a vocabulary-design feedback signal that this assessment level makes visible.

**Finding F5.** The logs are immediately consumable by standard tooling and support the analyses that motivated the pipeline (time budgets, material-handling shares, repetition structure, phase flows). End-to-end control-flow discovery is meaningful at the process level but not at the raw activity level; this is a property of flexibly ordered manual work as much as of the extraction, and the two-level design anticipates it (RQ3).

## Summary of Findings

- **F1 (L1).** Structurally sound logs at scale: 0 malformed events among 10,698; 0.33% residual boundary overlaps.
- **F2 (L2).** The closed-vocabulary contract is satisfied perfectly (100% conformance) with 89-97% catalogue utilization, no label hallucination, comparable logs.
- **F3 (L3).** Median temporal coverage ≈ 95% and boundary alignment 82-93% for videos up to one hour; a sharply characterized truncation failure on the seven multi-hour recordings.
- **F4 (L4).** Manual audit: strict event precision 0.86 with zero contradicted labels; operational recall ≈ 0.89, with misses dominated by the F3 mechanism.
- **F5 (L5).** High-fitness/low-precision discovery at activity level, substantially improved precision at process level; logs fit frequency/performance analysis directly and model discovery after abstraction.

Taken together, the assessment supports an affirmative answer to RQ1, a quantified and mostly positive answer to RQ2 with one well-localized failure mode, and a differentiated answer to RQ3 that ties analysis granularity to the vocabulary's two levels.

## Threats to Validity

**Construct validity.** L3 and L4 evaluate the log against the *transcript*, not against the video; errors of the vision stage (missed actions, hallucinated details, mis-scaled timestamps on long footage) are invisible to these levels and would propagate silently. The transcription prompt's uncertainty-marking requirement mitigates but does not eliminate this. A video-grounded audit (re-watching sampled intervals and judging both transcript and events) is the necessary next step and is planned as future work; the released per-video provenance (`source_file`, relative timestamps) makes such an audit directly executable.

**Internal validity.** The semantic audit was performed by a single author rather than by independent annotators, without inter-rater agreement; the samples (42+42) yield wide confidence intervals and are stratified by factory but not by video length, so the long-video regime is underrepresented in the precision sample. Boundary alignment uses a ±5 s tolerance; results are qualitatively stable under ±2 s but have not been reported per tolerance level.

**External validity.** Six factories from one corpus, all in labor-intensive discrete manufacturing, were processed; generalization to other domains (logistics, healthcare, construction) and to other foundation models is untested. The pipeline's behavior is prompt- and model-version-dependent; all prompts and model identifiers are released, but foundation-model APIs evolve, which limits exact reproducibility of the *generation* (the *assessment* of the released artifacts is fully reproducible from the repository).

**Conclusion validity.** Token-based replay was used for fitness/precision (alignment-based measures were computationally impractical at the activity level given trace uniqueness); absolute precision values should be compared across abstraction levels and factories, not across papers. The video-as-case notion treats each clip as an execution window, which fragments worker sessions; worker-as-case aggregation would change variant statistics.

## Reproducibility

All metrics in this section are computed by scripts operating solely on the released artifacts (per-video logs, transcripts, factory reports); the assessment code, the audit sample lists with verdicts, and fixed seeds are provided alongside the dataset. Re-running the assessment requires no access to the source videos or to any model API.
