# Approach

## Overview

The approach transforms raw egocentric factory video into process mining event logs through a pipeline of five fully automated stages. Each stage consumes the artifacts of the previous one and produces progressively more structured representations:

```
S1  video  ──────────────►  timestamped textual transcription        (per video)
S2  transcriptions ──────►  worker-level process summary             (per worker)
S3  worker summaries ────►  factory report + controlled vocabularies (per factory)
S4  transcription + vocab ─►  event list (JSON → CSV)                (per video)
S5  per-video CSVs ──────►  worker-level and factory-level event logs
```

The two central design principles are:

- **Late abstraction.** The video is first verbalized *without* any process mining schema (S1); activity labels are introduced only after a factory-wide vocabulary has been established (S3) and are then applied retrospectively (S4). This avoids committing to an event granularity before the overall behavior of the factory is known, and it mirrors the extraction-correlation-abstraction decomposition of event-data preparation [DIBA2020].
- **Closed-world labeling.** Event extraction (S4) may only use activity and process labels from the factory vocabularies constructed in S3. The vocabularies act as a factory-specific abstraction layer that suppresses label hallucination, harmonizes granularity across videos and workers, and makes the resulting logs comparable and minable.

All stages are implemented as Python scripts that call foundation models through REST APIs; no model is fine-tuned, and no human annotation enters the pipeline at any point. Prompts are stored as versioned plain-text files alongside the code.

## Source Data

The input is the Egocentric-10K corpus, a large-scale dataset of first-person factory videos recorded with head-mounted cameras by workers in real industrial facilities, organized as `factory_xxx/workers/worker_xxx/*.tar` archives of video clips. We processed six factories covering heterogeneous domains of labor-intensive manufacturing: metal stamping and mechanical assembly (factory 001), garment finishing and packaging (factories 002 and 004), metal casting post-processing (factory 003), machining and motor assembly (factory 005), and sand-mold foundry work (factory 006). The processed subset comprises 59 workers and 322 video clips. Clip length is typically around 20 minutes; a small number of recordings span multiple hours.

## S1: Change-Point Transcription of Videos

Each video is uploaded to a multimodal foundation model (Gemini 3.1 Pro in our instantiation) together with a transcription prompt that requests a *detailed chronological narration* consisting of entries of the form

```
[HH:MM:SS - HH:MM:SS] free-text description of observable behavior
```

The prompt encodes three requirements that matter for downstream event extraction:

1. **Change-point segmentation.** A new entry must be opened when (and only when) something operationally meaningful changes: the action, the handled object, the location, the focus of attention, the involvement of another person, or the occurrence of an anomaly. Stable repetitions ("the wearer continues wiping the same section of the table") must be merged into one long interval instead of being split into near-identical micro-entries. Transcript boundaries thereby approximate behavioral change points, which S4 later reuses as candidate event boundaries.
2. **Descriptive neutrality.** The model is explicitly forbidden to output labels, categories, tables, JSON, or analysis language. This prevents the vision model from prematurely imposing an ad-hoc activity schema and keeps the transcription reusable under different, later-defined vocabularies.
3. **Marked uncertainty.** When the scene is unclear (motion blur, occlusion, camera removal), the model must state the uncertainty briefly rather than guess.

Across the six factories, S1 produced 13,304 timestamped entries for the 322 videos. The stage includes operational safeguards relevant for reproducibility at scale: resumable execution (existing transcriptions are never overwritten), exponential-backoff retries on transient API errors, and deletion of uploaded video files after processing.

## S2: Worker-Level Process Summaries

For each worker, all transcriptions are synthesized into a structured plain-text report with fixed sections (*Process Summary*, *Detailed Process Breakdown*, *Possible Process Optimizations*, *Evidence Limits*). The prompt directs the model to identify the distinct processes the worker performs, to describe their operational flow (setup, repeated action loops, machine interaction, material handling, inspection, transport, handoffs), and to restrict optimization remarks to process-level design issues, explicitly excluding judgments about individual worker performance, both for methodological and for ethical reasons. The *Evidence Limits* section forces an explicit statement of what could not be established from the footage, propagating S1's uncertainty marking to the aggregate level.

The worker summaries serve two purposes: they are an analysis product in their own right, and they are the evidence base from which factory-level vocabularies are induced in S3, a deliberate bottleneck that prevents idiosyncrasies of single videos from contaminating the factory-wide label set.

## S3: Factory Reports and Controlled Vocabularies

The worker summaries of each factory are aggregated into a factory-level report with fixed sections, two of which constitute the controlled vocabularies used by S4:

- **Factory Process Labels**: a deduplicated list of coarse process labels (12-24 per factory in our corpus), e.g., *Metal stamping*, *Material replenishment*, *Sand mold creation*. Process labels partition the work into phases comparable to sub-processes in a business-process architecture.
- **Factory Activity Catalogue**: a rich list of fine-grained activity labels (55-131 per factory), each a short verb-object phrase, e.g., *load raw metal blank into die*, *wrap white cardboard band around folded shirt*, *level powder surface with wooden scraper*. The prompt requires specific verb-object phrasing (rejecting generic labels such as *operate machine*), demands coverage of non-value-adding behavior (transport, staging, waiting, setup, cleanup, inspection, handoffs, documentation), and prohibits near-duplicate labels.

The catalogue is intentionally biased toward *optimization-relevant* distinctions: labels are preferred if they help reveal bottlenecks, queues, machine idle time, material-handling burden, batching, rework, and coordination dependencies. In terms of the event-abstraction literature [VANZELST2021], S3 defines the abstraction target onto which the low-level observations of S1 are lifted; the two-level structure (process ⊃ activity) provides two analysis granularities from the same log.

## S4: Constrained Event Extraction

Each raw transcription is converted, by a text-only LLM (GPT-5.4 in our instantiation), independently per video and parallelized across videos, into a JSON object containing a list of events with exactly four fields: `estimated_start_time`, `estimated_end_time`, `activity`, `process`. The prompt supplies the factory's two vocabularies and the worker summary as context and imposes the following contract:

- **Closed vocabularies.** `activity` and `process` must be *exactly* one of the allowed labels; inventing labels is forbidden; if several labels are plausible, the closest allowed one must be chosen.
- **Transcript-anchored timing.** Transcript timestamps are the primary basis for event boundaries. A transcript interval containing one stable repeated activity becomes one event; an interval that clearly contains multiple different actions may be split, with estimated internal boundaries.
- **Operational relevance filter.** Only process-relevant operational events may be emitted; incidental camera motion and non-work segments are to be ignored. An empty event list is an admissible output.
- **Strict output discipline.** JSON only, fixed schema; outputs are validated, and label conformance is checked against the vocabularies at parse time.

The result is one CSV per video with columns `start_timestamp`, `end_timestamp`, `activity`, `process`, `factory`, `worker`, where timestamps are rendered on a synthetic time axis anchored at 2000-01-01 00:00:00 plus the offset from video start (videos carry no wall-clock metadata; see the discussion of timestamp semantics below).

## S5: Log Consolidation

Per-video CSVs are merged into worker-level logs and factory-level logs. Because the source clips of one worker are consecutive segments of recording sessions, the merge concatenates a worker's per-video logs in clip order and shifts each clip's relative timestamps by the cumulative end time of the preceding clips, producing a single continuous synthetic timeline per worker. Each merged event carries the additional attributes `source_file` (provenance: the originating per-video log), `case_id` (set to `factory_worker`), `start_seconds_from_video_start`, `end_seconds_from_video_start`, and `duration_seconds`. Factory-level logs are the union of the factory's worker-level logs.

The released artifacts thus support at least three case notions without re-extraction: *video-as-case* (each clip is one observed execution window; used in our assessment), *worker-as-case* (one long trace per worker, suitable for session-level analysis), and (via the `process` attribute) *process-phase segmentation* within either notion. The choice of case notion is a modeling decision, not a property of the data [DEMURILLAS2020], and the flat CSV deliberately retains all attributes needed to re-derive alternatives.

## Design Decisions and Rationale

**Why text as the intermediate representation?** Direct video-to-event log extraction would couple the (expensive, per-video) vision step to the (factory-global) vocabulary, so any vocabulary revision would require re-processing all videos. The textual transcription decouples the two: it is created once, is human-auditable, and serves simultaneously as extraction source (S4), evidence base for vocabulary induction (S2-S3), and reference for the fidelity assessment. The transcript is, in effect, the pipeline's provenance layer.

**Why factory-specific rather than global vocabularies?** Manual work is domain-specific; a foundry and a garment factory share almost no activities. Factory-scoped vocabularies keep labels concrete (verb-object with domain nouns) while still harmonizing across workers of the same factory, the level at which comparative process analysis is actually performed. Cross-factory comparability is retained at the coarse process level, where labels such as *quality inspection* or *material replenishment* recur.

**Timestamp semantics.** All timestamps are *relative to video start* and rendered on a synthetic absolute axis for tool compatibility. Within-video durations and orderings are meaningful; absolute dates are not, and gaps between recording sessions are not represented. Performance analyses on the merged logs must therefore be interpreted as *observed-work-time* analyses rather than calendar-time analyses.

**Event boundaries under repetition.** Egocentric factory work is dominated by short-cycle repetition. Following the anti-repetition rule of S1, a stretch of dozens of identical cycles is represented as one long event rather than dozens of two-second events. This is a conscious abstraction trade-off: it makes logs readable and models discoverable, at the cost of hiding intra-stretch cycle counts. Cycle-level analysis remains possible where the transcription resolved individual cycles, and the assessment quantifies the resulting duration distributions.

## Implementation

The pipeline is implemented as five sequentially numbered Python scripts (plus utilities) that communicate exclusively through the file system, using one directory tree per artifact layer (`raw_transcriptions/`, `worker_process_summaries/`, `factory_process_mining_reports/`, `process_mining_event_logs/`, `merged_process_mining_event_logs/`). S1-S3 call the Gemini REST API (model `gemini-3.1-pro-preview`); S4 calls the OpenAI Responses API (model `gpt-5.4`) with up to 100 concurrent requests. All stages are idempotent (existing outputs are kept unless overwriting is requested), retry transient HTTP failures with exponential backoff, and validate outputs before writing. Prompts, scripts, and all derived artifacts for the six factories are publicly available in the project repository.
