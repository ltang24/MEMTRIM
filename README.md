# MEMTRIM

<p align="center">
  <img src="assets/memtrim_intro.png" width="95%" alt="Memory overreliance example">
  <br>
  <em>A retrieved memory whose evidence only partly applies to the current query.</em>
</p>

<p align="center">
  <img src="assets/memtrim_method.png" width="98%" alt="MEMTRIM overview">
  <br>
  <em>Overview of the MEMTRIM write-time index and read-time filtering.</em>
</p>

MEMTRIM is a layer placed around an existing agent memory system that indexes each stored entry as evidence units in a trie at write time. At read time it drops retrieved evidence the query already states or contradicts and suppresses the stored answer when its support or the task has changed. MEMTRIM parses and indexes exactly the entries each memory system stores and returns, so it works on top of Mem0, A-Mem, MemGPT (Letta) and Graphiti without changing their write or retrieval paths.

The main experiment builds task-aligned (TA) and task-divergent with overlapping evidence (TD-Dup) items from three datasets, writes them into stores of four memory systems, and answers every item with each baseline and with MEMTRIM. Every method for a given item and memory system sees the same retrieved entries.

## Installation

Requires Python 3.11+ and a CUDA 12.1 compatible PyTorch build (see `requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

The script downloads RippleEdits, CounterLogic and CLADDER at pinned commits into `RippleEdits/`, `CounterLogic/` and `cladder/` and checks the files' SHA-256 sums.

## Servers

The pipeline talks to any OpenAI-compatible chat endpoint that reports `max_model_len` on `/v1/models`. For example, with vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name qwen --port 8000
```

MemGPT needs a running Letta server (see the Letta documentation), reachable at `MEMTRIM_LETTA_BASE_URL`, with a model and an embedding provider configured whose handles match `MEMTRIM_LETTA_MODEL` and `MEMTRIM_LETTA_EMBEDDING`.

Graphiti runs embedded on Kuzu by default and needs no server. Set `MEMTRIM_GRAPHITI_URI` to use Neo4j instead.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MEMTRIM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `MEMTRIM_MODEL` | `qwen` | Model name served by the endpoint |
| `MEMTRIM_MODEL_HF_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face tokenizer used to check prompt lengths |
| `MEMTRIM_ENCODER_DEVICE` | `cpu` | Device for the sentence encoder and compressors, e.g. `cuda` |
| `MEMTRIM_LETTA_BASE_URL` | `http://localhost:8283` | Letta server used by the MemGPT backend |
| `MEMTRIM_LETTA_MODEL` | `openai/$MEMTRIM_MODEL` | Letta model handle for the MemGPT agents |
| `MEMTRIM_LETTA_EMBEDDING` | `openai/text-embedding-3-small` | Letta embedding handle for archival memory |
| `MEMTRIM_GRAPHITI_URI` | empty (Kuzu) | Neo4j URI for Graphiti |
| `MEMTRIM_GRAPHITI_USER`, `MEMTRIM_GRAPHITI_PASSWORD` | `neo4j`, empty | Neo4j credentials |

## Running

One command line covers every dataset, memory system and method:

```bash
PYTHONPATH=src python -m memtrim.experiments.main build-data
PYTHONPATH=src python -m memtrim.experiments.main smoke
PYTHONPATH=src python -m memtrim.experiments.main run
```

- `build-data` builds the items and stores of each dataset and writes them to `data/<dataset>/items.json` and `data/<dataset>/stores.json`. It needs neither the LLM nor a memory system.
- `smoke` runs the full pipeline on 5 items per condition.
- `run` runs the full pipeline with the settings in `configs/main.yaml`.

`run` and `smoke` read `data/<dataset>/items.json` and `stores.json` when they were built with the same seed, `n_per_condition`, `max_store_size` and dataset parameters, and build the dataset otherwise. All datasets go through the same code path and the same answering prompt.

Options shared by all subcommands:

| Option | Meaning |
|---|---|
| `--config` | Configuration file, default `configs/main.yaml` |
| `--datasets` | Subset of the datasets listed in the configuration |
| `--memories` | Subset of the memory systems |
| `--methods` | Subset of the methods |
| `--n` | Items per condition (overrides `n_per_condition`) |
| `--seed` | Seed (overrides `seeds`) |

```bash
PYTHONPATH=src python -m memtrim.experiments.main run --datasets cladder --memories mem0 graphiti --methods no_memory base_memory memtrim
```

Predictions are appended to `data/results/{run,smoke}_seed{SEED}.jsonl`, one line per dataset, item, memory system and method. Re-running the same command skips everything already in that file, so an interrupted run resumes where it stopped. The plain-text summary is printed at the end and can be printed again with:

```bash
PYTHONPATH=src python -m memtrim.evaluation.report data/results/run_seed42.jsonl
```

It reports TA accuracy, TD-Dup accuracy, and on TD-Dup the share of items that flip from wrong to correct and from correct to wrong relative to answering without memory.

## Datasets

| Name | Source | TA / TD-Dup | Retrieval group |
|---|---|---|---|
| `rippleedits` | RippleEdits | Stored interaction states the new fact (TA) or the original fact with its old answer (TD-Dup); the query states the new fact and asks the alias question | Relation |
| `counterlogic` | CounterLogic `Dataset/data_v3.json` | Stored interaction asks whether the argument is logically valid (answer: validity); the query asks whether the conclusion agrees with real-world knowledge (gold: believability). TA when validity equals believability, TD-Dup otherwise | Agglomerative clusters of premise and conclusion embeddings (cosine, average linkage, distance threshold 0.2, clusters with at least 5 instances) |
| `cladder` | CLADDER `cladder-v1-q-balanced.json` with backgrounds from `cladder-v1-meta-models.json` | A source question with its answer is stored; the query is a paired question whose given numbers partly change. TA when both answers agree, TD-Dup otherwise | Story, graph, query type and rung |

CLADDER pairs are formed within a retrieval group when the number-masked `given_info` texts have similarity of at least 0.85, the `given_info` keys have Jaccard overlap of at least 0.8 with at least one unchanged and one changed value, the question text is identical, and a changed value appears in the candidate's reasoning. All datasets are built deterministically without an LLM, sample up to `n_per_condition` items per condition with each seed, and fill stores of up to `max_store_size` interactions per retrieval group and condition. RippleEdits items additionally store up to three unchanged facts about the same subject as separate interactions.

## Memory systems

| Name | Write path | Retrieval |
|---|---|---|
| `mem0` | `Memory.add` (FAISS store, local sentence encoder) | `Memory.search` over extracted memories |
| `amem` | `AgenticMemorySystem.add_note` | `search_agentic` over notes |
| `memgpt` | One Letta agent per store; every interaction written to archival memory tagged with its interaction id, and the most recent one held in a core memory block | Core block first, then archival passage search |
| `graphiti` | One `group_id` per store, one episode per interaction | Hybrid search over relation facts, each linked back to its source episode |

## Methods

| Name | Context given to the answering prompt |
|---|---|
| `no_memory` | None |
| `base_memory` | The top-k retrieved entries |
| `facts_only` | Each retrieved entry replaced by its source interaction without the stored answer |
| `top1` | Only the first retrieved entry |
| `mmr` | Maximal marginal relevance (lambda 0.5) over a pool of 15 retrieved entries |
| `sim_filter` | Retrieved entries with cosine similarity to the query of at least 0.5 |
| `recomp` | Top 3 sentences by query similarity with the `fangyuan/nq_extractive_compressor` extractive compressor |
| `llmlingua2` | Retrieved entries compressed by `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` at rate 0.5 |
| `context_faithful` | Retrieved entries, with an instruction to treat the query as ground truth and ignore memories that disagree |
| `answer_first` | Answer from the query alone, then revise only if the memories are consistent with the query |
| `micro_act` | An explicit step listing same, conflicting and absent facts between memories and query, then the answer |
| `memtrim` | The MEMTRIM-filtered context built from the retrieved entries |

All methods use the same answering prompt template. MEMTRIM ablations are set in `configs/main.yaml` under `memtrim`: `keep_shared`, `keep_conflicts`, `keep_old_answer`, `ignore_task_change`, `embedding_match`.

## Module map

```
src/memtrim/
  core/          config, schema, trie, read-time filtering and multi-memory merge (readtime.py),
                 parser, canonicalisation and equivalence (alias.py, equivalence.py), prompts,
                 LLM client with SQLite cache, sentence encoder, answer matching
  datasets/      base.py (Item, Interaction, sampling, store building), registry.py,
                 rippleedits.py, counterlogic.py, cladder.py
  memory/        base.py (MemoryBackend: add, search, entries, close), registry.py,
                 mem0.py, amem.py, memgpt.py, graphiti.py
  baselines/     base.py (Method.run(item, retrieved_entries, llm)), registry.py, one file per method
  memtrim_method.py   MEMTRIM as a Method, indexing each store's entries
  evaluation/    metrics.py, report.py
  experiments/   main.py: build-data, smoke and run over datasets x memory systems x methods
configs/main.yaml
scripts/download_data.sh
data/<dataset>/   items.json, stores.json
tests/
```

## Tests

Unit tests need no LLM, server or model download:

```bash
python -m unittest discover -s tests -v
```

## Data

| Dataset | Folder | Source repository | Commit | License | In this repository |
|---|---|---|---|---|---|
| RippleEdits | `data/rippleedits/` | [edenbiran/RippleEdits](https://github.com/edenbiran/RippleEdits) | `54f3b88af4895a3aacb580ec63ce7ae857185040` | MIT (`LICENSE.txt`, copied to `data/rippleedits/LICENSE-rippleedits.txt`) | included |
| CLADDER | `data/cladder/` | [causalNLP/cladder](https://github.com/causalNLP/cladder) | `3d2d1169b4b939a09048a6a75956c8972a93cc38` | MIT (`LICENSE`, copied to `data/cladder/LICENSE-cladder.txt`) | included |
| CounterLogic | `data/counterlogic/` | [Sukin272/CounterLogic](https://github.com/Sukin272/CounterLogic) | `e35cd1125667f2c8a1596085750fecb1a4b0a0de` | no license file in the source repository; downloaded and built locally | built locally by `build-data` |

The included folders hold `items.json` and `stores.json` built by `build-data --n 500 --seed 42` with the settings in `configs/main.yaml`; running the same command on the sources fetched by `scripts/download_data.sh` produces the same files. Raw source files are not redistributed.

## Outputs

| Path | Written by |
|---|---|
| `data/<dataset>/items.json`, `data/<dataset>/stores.json` | `build-data` |
| `data/results/run_seed{SEED}.jsonl` | `run`: one line per dataset, item, memory system and method |
| `data/results/smoke_seed{SEED}.jsonl` | `smoke` |
| `data/llm_cache.sqlite3` | all LLM calls are cached here |
| `data/mem0_stores/` | Mem0 FAISS stores and A-Mem Chroma stores |
| `data/graphiti_stores/` | Graphiti Kuzu databases |
| `logs/errors.jsonl` | memory write and retrieval failures |
