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

MEMTRIM is a layer placed around an existing agent memory system (here Mem0 and A-Mem) that indexes each stored interaction as evidence units in a trie at write time. At read time it drops retrieved evidence the query already states or contradicts and suppresses the stored answer when its support or the task has changed. This pipeline builds task-aligned (TA) and task-duplicated (TD-Dup) items from RippleEdits, writes them into Mem0 and A-Mem stores, and answers each item with no memory, with the raw retrieved memories, and with MEMTRIM.

## Installation

Requires Python 3.11+ and a CUDA 12.1 compatible PyTorch build (see `requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_rippleedits.sh
```

The script downloads the RippleEdits benchmark at a pinned commit into `RippleEdits/` and checks the files' SHA-256 sums.

## LLM server

The pipeline talks to any OpenAI-compatible chat endpoint that reports `max_model_len` on `/v1/models`. For example, with vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name qwen --port 8000
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MEMTRIM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `MEMTRIM_MODEL` | `qwen` | Model name served by the endpoint |
| `MEMTRIM_MODEL_HF_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face tokenizer used to check prompt lengths |
| `MEMTRIM_ENCODER_DEVICE` | `cpu` | Device for the sentence encoder, e.g. `cuda:0` |

## Usage

Unit tests (no server needed):

```bash
python -m unittest discover -s tests -v
```

Quick check on 5 edits:

```bash
PYTHONPATH=src python -m memtrim.cli smoke --n 5
```

Full run (200 edits, seed 42):

```bash
PYTHONPATH=src python -m memtrim.cli run --n 200 --seed 42
```

Build only the items and stores, without calling the LLM:

```bash
PYTHONPATH=src python -m memtrim.cli build-data --n 200 --seed 42
```

Common options: `--top-k`, `--max-store-size`, `--workers`, and the ablation switches `--keep-shared`, `--keep-conflicts`, `--keep-old-answer`, `--ignore-task-change`, `--embedding-match`.

## Data

`data/items.json` and `data/stores.json` are the items and memory stores built by `build-data --n 200 --seed 42`. They are derived from RippleEdits, which is distributed under the MIT License (see `data/LICENSE-RippleEdits.txt`).

## Outputs

| Path | Written by |
|---|---|
| `data/items.json`, `data/stores.json` | `build-data`, `run` (overwritten with the items built for the given `--n` and `--seed`) |
| `data/results/predictions_n{N}_seed{SEED}.jsonl` | `run`: one line per item, method and memory system |
| `data/results/smoke_predictions.jsonl` | `smoke` |
| `data/llm_cache.sqlite3` | all LLM calls are cached here |
| `data/mem0_stores/` | Mem0 FAISS stores and A-Mem Chroma stores |
| `logs/errors.jsonl` | memory write and retrieval failures |

The summary table with TA and TD-Dup accuracy and flip rates is printed to stdout.
