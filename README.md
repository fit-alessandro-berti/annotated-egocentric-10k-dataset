# Annotated Egocentric-10K Dataset

This project contains a derived annotation and process-mining pipeline built on top of the [Egocentric-10K dataset](https://huggingface.co/datasets/builddotai/Egocentric-10K), a large-scale first-person factory video dataset collected from real industrial work.

The goal of this repository is not to redistribute the source dataset itself, but to transform selected factory/worker video clips into progressively more structured artifacts that are useful for process understanding, operations analysis, and process mining.

## Project Purpose

The repository is designed to turn raw egocentric factory video into several layers of machine-generated process documentation:

1. detailed chronological text transcriptions of worker videos
2. worker-level process summaries
3. factory-level process summaries, process labels, and process-mining activity vocabularies
4. per-video CSV event logs aligned to the factory vocabulary
5. merged sequential event logs per worker and per factory

In practice, the project is meant to support downstream analysis of:

- what workers are doing
- which processes are present in a factory
- which fine-grained activities are good candidates for process mining
- where material handling, waiting, batching, handoffs, or workstation-layout issues may exist

## Source Dataset

The source data comes from:

- [Egocentric-10K on Hugging Face](https://huggingface.co/datasets/builddotai/Egocentric-10K)

At the time of writing, the dataset card describes Egocentric-10K as a large egocentric factory dataset with long-duration real-world industrial footage, organized by `factory_xxx/workers/worker_xxx/*.tar`.

## What Has Been Done In This Repository

This repository contains scripts and prompt templates to build a structured annotation pipeline on top of Egocentric-10K.

The work done here includes:

- generating detailed text transcriptions from raw factory videos
- saving these transcriptions under `raw_transcriptions/factory_xxx/worker_xxx/*.txt`
- summarizing each worker's observed processes into worker-level reports
- aggregating worker summaries into factory-level process-mining reports
- extracting a factory-specific process label list and factory-specific activity label list
- converting each raw annotation into a CSV process-mining event log constrained by the factory vocabulary
- merging per-video CSV logs into worker-level ordered logs
- concatenating worker-level logs into factory-level logs

The repository therefore contains both:

- code for generating the derived artifacts
- previously generated outputs for part of the dataset

## Logical Execution Order

The main processing scripts in this repository are named in execution order:

1. [01_transcribe_factory.py](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/01_transcribe_factory.py)
2. [02_summarize_worker_processes.py](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/02_summarize_worker_processes.py)
3. [03_summarize_factory_process_mining.py](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/03_summarize_factory_process_mining.py)
4. [04_annotation_to_event_log.py](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/04_annotation_to_event_log.py)
5. [05_merge_event_log_csvs.py](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/05_merge_event_log_csvs.py)

Auxiliary cleanup utility:

- [delete_uploaded_files.py](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/delete_uploaded_files.py)

## Prompt Files

The prompt templates used by the pipeline are stored as plain text files:

- [annotation_prompt.txt](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/annotation_prompt.txt)
- [worker_process_summary_prompt.txt](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/worker_process_summary_prompt.txt)
- [factory_process_mining_prompt.txt](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/factory_process_mining_prompt.txt)
- [annotation_to_event_log_prompt.txt](/mnt/c/Users/berti/annotated-egocentric-10k-dataset/annotation_to_event_log_prompt.txt)

## Output Structure

Derived outputs are organized by factory and worker.

Typical directories are:

- `raw_transcriptions/`
- `worker_process_summaries/`
- `factory_process_mining_reports/`
- `process_mining_event_logs/`
- `merged_process_mining_event_logs/`

The main naming convention throughout the repository is:

- factory: `factory_xxx`
- worker: `worker_xxx`
- per-video artifacts: one file per original video clip

## Notes

- This repository contains derived annotations and analysis artifacts, not the full original Egocentric-10K dataset payload.
- Some stages rely on external LLM APIs and prompt files.
- Factory-level vocabularies are used to constrain downstream event-log generation, so later stages depend on earlier stages having been run first.
