# MEMTRIM

<p align="center">
  <strong>Mitigating inference-time overreliance in agentic memory</strong>
</p>

An anonymous research-code release accompanying an ICLR submission, providing a
compact reference implementation of the core method and a synthetic RQ1 example.

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#repository-structure">Repository Structure</a>
</p>

## Overview

Memory can be highly relevant to the current query while still carrying
context-specific information that should not be reused. Shared entities or
circumstances alone do not establish that a past conclusion still applies.

<p align="center">
  <img src="assets/memtrim_intro.png" alt="Relevant retrieved memory can carry context-specific information that misleads the current query." width="95%">
</p>

MEMTRIM is a plug-and-play layer around an existing agentic memory system. It
transforms retrieved memories before context construction, giving current-query
evidence priority. The underlying retriever remains responsible for retrieval.

## Method

<p align="center">
  <img src="assets/memtrim_method.png" alt="MEMTRIM write-time evidence indexing and read-time evidence partitioning, outcome handling, and context construction." width="98%">
</p>

**Write time:** memory → evidence units + task signature + support evidence +
stored outcome → trie-backed evidence index.

**Read time:** retrieved memory + current query → shared / conflicting /
memory-only evidence → context construction.

A memory is represented as `m = (E_m, tau_m, H_m, o_m)`: canonical key-value
evidence `E_m`, task signature `tau_m`, support evidence `H_m` (a subset of `E_m`),
and stored outcome `o_m`. Each evidence unit is indexed independently under
canonical key → value → memory IDs. This minimal implementation accepts already
structured records and memory IDs supplied by an external retriever.

| Component | MEMTRIM behavior |
| --- | --- |
| Shared evidence | Do not repeat |
| Conflicting evidence | Current query takes precedence |
| Memory-only evidence | Retain |
| Stored outcome | Suppress if supporting evidence conflicts |

Shared evidence matches the query's key and value; conflicting evidence has the
same key but a different value; memory-only keys are absent from the query.
Conflicting memory evidence is removed. A conflict on a support key suppresses
both the stored outcome and its evidence representation. Otherwise, matching
task signatures retain the outcome; task-mismatched outcomes can be reused as
evidence only through an explicitly supplied typed `outcome_evidence` representation.

Context construction emits identical evidence once and preserves multiple values
from one memory. If different memories provide inconsistent value sets for a key
absent from the query, all values for that key are withheld. Ordering is deterministic.

## Quick Start

Python 3.10+ is recommended. The implementation uses only the Python standard
library; no package installation is required, and `requirements.txt` is empty.
Replace the placeholder with the anonymous repository URL:

```sh
git clone <ANONYMOUS_REPOSITORY_URL>
cd MEMTRIM
```

Run the demo from the repository root:

```sh
python scripts/demo.py
```

The demo prints the evidence partitions, outcome decisions, and final contexts
for two cases:

- **Match:** supporting evidence remains unchanged, so the stored outcome is retained.
- **Mismatch:** answer-relevant supporting evidence changes, so the previous outcome is suppressed.

## Evaluation

### Minimal RQ1 evaluation

```sh
python scripts/run_rq1_minimal.py
```

This runs a small synthetic example of the RQ1 evaluation protocol using saved
predictions, without LLM inference. It reports separately for match and mismatch:

- No-memory accuracy and memory accuracy.
- Correct-to-wrong and wrong-to-correct transition counts and rates.

Accuracy uses exact equality with the gold answer. Each transition rate is its
count divided by the total number of examples in that split.

The JSONL examples are synthetic and are provided only to illustrate the
evaluation code and metric definitions. They are not paper results and are
not intended to reproduce Table 1.

## Repository Structure

```text
MEMTRIM/
├── .gitignore                # Local artifacts and environment files
├── README.md                 # Method and usage
├── requirements.txt          # Empty; standard library only
├── assets/
│   ├── memtrim_intro.png      # Motivation figure
│   └── memtrim_method.png     # Method figure
├── memtrim/
│   ├── __init__.py            # Public package exports
│   ├── schema.py              # Evidence, memory, query, and result dataclasses
│   ├── trie.py                # Evidence-level key/value index
│   └── core.py                # Memory transformation and context merging
├── scripts/
│   ├── demo.py                # Deterministic match/mismatch demonstration
│   └── run_rq1_minimal.py     # Synthetic RQ1 metric evaluation
├── data/
│   └── rq1_toy.jsonl          # Eight synthetic prediction examples
└── tests/
    └── test_context_merge.py  # Context-merging regression checks
```

.
