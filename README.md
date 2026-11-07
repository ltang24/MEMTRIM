# MEMTRIM

<p align="center">
  <b>Understanding and Mitigating Inference-Time Overreliance Using Agentic Memory</b>
</p>

<p align="center">
  <img src="assets/memtrim_intro.png" width="95%" alt="Illustration of memory overreliance">
</p>

A retrieved memory can be benign, correctly stored and correctly retrieved, yet
still mislead the model when only part of its evidence transfers to the current
query (**memory overreliance**). MEMTRIM is a plug-and-play layer around an
existing memory system: it indexes each stored interaction as canonical
evidence units in a trie at write time, and at read time removes evidence the
query already states, drops evidence that conflicts with the query, keeps
memory-only evidence, and suppresses / relabels the stored answer when its
support changed or the task changed (Eq. 1).

<p align="center">
  <img src="assets/memtrim_method.png" width="98%" alt="Overview of MEMTRIM">
</p>

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Servers you need for the full runs:

| What | How |
|---|---|
| Open-weight backbone | `vllm serve <checkpoint> --port 8000 --served-model-name qwen` (set `llm.model` / `base_url` in `configs/main_qwen.yaml`) |
| API backbone | `export OPENAI_API_KEY=...` (set `llm.model` in `configs/main_gpt.yaml`) |
| MemGPT | `pip install letta && letta server` (port 8283) |
| Graphiti | nothing (embedded Kuzu); or set `memory.graphiti.uri` for Neo4j |

## Quick check (offline, ~1 s)

```bash
python -m unittest discover -s tests            # 24 tests: trie, Eq. 1, ablations, parser, case studies, stats
python -m memtrim.data.build --encoder hash --out-dir data/processed_smoke   # needs data/raw, see below
python scripts/run.py --config configs/smoke.yaml && python -m memtrim.report results/smoke
```

`configs/smoke.yaml` uses a mock LLM and a local dense store; it checks that
every stage runs, and its numbers mean nothing.

## Reproducing the paper

```bash
bash scripts/download_data.sh                   # RippleEdits, CLADDER, CounterLogic at pinned commits
python -m memtrim.data.build --dataset all      # Appendix C -> data/processed/*.jsonl (+ balance report, C.4)
bash scripts/run_all.sh                         # everything below, resumable
```

Before a full run, try any config with `--limit 5` to confirm your servers work.

| Paper | Command | Output |
|---|---|---|
| Table 1 (RQ1) | `scripts/run.py --config configs/main_{qwen,gpt}.yaml` → `memtrim.report` | rows `No Mem.` / backbone in `results/<run>/tables.md` |
| Table 2 (Mem0) / Tables 3–4 | same runs (all baselines × 4 backbones) | `tables.md`, `tables.tex`, `summary.csv` |
| Fig. 2 / Fig. 8 (RQ2) | `python -m memtrim.experiments.rq2 results/<run>` | `fig_rq2_<dataset>.pdf`, `rq2_<dataset>.json` |
| Fig. 3 / Fig. 9 (RQ3) | `python -m memtrim.experiments.rq3 --config configs/rq3.yaml` | `fig_rq3_<dataset>.pdf`, `rq3_report.json` |
| Fig. 5 (ablations) | `scripts/run.py --config configs/ablation_qwen.yaml` → `report --ablation` | `fig5_ablation.pdf` |
| Fig. 6 (attacks) | `python -m memtrim.experiments.attack --config configs/attack.yaml` | `fig6_attack.pdf` |
| Fig. 7 (runtime) | `scripts/run.py --config configs/runtime_qwen.yaml` → `report --runtime` | `fig7_runtime.pdf` |
| App. B.4 case studies | `tests/test_method.py::CaseStudies`, `Eq1Tests` | – |

All LLM calls are cached on disk (`.cache/llm`), so re-running a config
reproduces the same outputs without re-querying; the runtime config bypasses
the cache.

## Code map

```text
memtrim/
  schema.py      EvidenceUnit, MemoryRecord m=(E_m, tau_m, H_m, o_m), MemTrimOptions (ablation switches)
  trie.py        trie index: key path (entity|attribute|type) -> value leaf -> {memory_id: in_support}
  core.py        write-time indexing; read-time partition S/D/U, Eq. 1, multi-memory merge, rendering
  parser.py      LLM parser: typed units, span grounding, validation + 1 retry, key/task/value canonicalisation
  llm.py         OpenAI-compatible client (vLLM or OpenAI), deterministic decoding, sqlite cache; mock backend
  memory/        adapters: Mem0, A-Mem, MemGPT (Letta), Graphiti; local dense store for tests
  methods.py     no-memory, backbone, 9 baselines, MEMTRIM + 5 ablations, 3 attack defenses
  data/build.py  Appendix C pair construction (no LLM involved)
  runner.py      stores, retrieval, all methods on the same retrieved items, resumable JSONL output
  report.py      Tables 1-4, Fig. 5, Fig. 7
  stats.py       Simonsohn two-lines test, quadratic test, RQ2 overlap tests
  experiments/   rq2.py, rq3.py, attack.py
```

## Implementation choices not fixed by the paper

These are the settings this code uses where the paper text does not pin a
value. They are all in the configs; report them (or change them to what was
originally used).

* **Memory stores.** One store per (condition, retrieval group): RippleEdits
  grouped by relation, CounterLogic by the C.2 clusters, CLADDER by story.
  Each store holds ≤ 20 stored interactions; other items' memories act as
  distractors. `top_k = 5` for main runs, `k = 16` for attacks.
* **Seeds** change memory write order / store composition and the LLM seed;
  decoding is greedy (temperature 0).
* **RippleEdits interactions.** The current query states the edited fact and
  asks the Subject-Aliasing question; stored interactions additionally carry up
  to three unchanged facts about the subject (Relation-Specificity /
  Forgetfulness), which provide memory-only evidence.
* **Facts-only** replaces each retrieved item by its source interaction without
  the stored answer.
* **Baselines and defenses** are re-implementations from the papers'
  descriptions (prompts in `methods.py`; RECOMP extractive compressor
  `fangyuan/nq_extractive_compressor`, top-3 sentences; LLMLingua-2 rate 0.5;
  MMR λ = 0.5; similarity filter τ = 0.5; RobustRAG α = 0.3; TrustRAG τ = 0.88).
  Attacks are likewise simplified re-implementations (see `experiments/attack.py`).
* **MEMTRIM details beyond App. A.1:** disagreeing stored outcomes from several
  retrieved memories are withheld (same rule as conflicting evidence); a memory
  whose parse fails contributes its raw text without its answer
  (`on_parse_failure`); Graphiti search uses its own hybrid ranking (no
  cross-encoder).
* **RQ2 statistics:** Mann–Whitney U, a one-sided test that failures lie closer
  to the centre of the overlap range, and the two-lines test on
  P(failure | overlap). **RQ3:** units are copied non-support first, support last;
  the TA question/answer applies once every support unit is restored.

## Things to check against the paper before reporting

Found while building the data from the public releases:

* **CounterLogic** (`Dataset/data_v3.json`, the released version with labels)
  has **1,150** instances, not 1,800. TA/TD splits: 547 / 603.
* **RippleEdits** RANDOM + POPULAR contain **2,807** edits, not 2,000.
* **CLADDER-Derived.** The App. C.3 rules as written give 500 pairs only if the
  paired question may differ (e.g. "positively" vs "negatively"). Requiring that
  only causal values change (Table 5) gives far fewer pairs; this is the
  `--cladder-same-question` flag.
* **RippleEdits TA vs TD-Dup no-memory accuracy.** Under C.1 both conditions use
  the same current queries, so `No Mem.` accuracy is identical for TA and
  TD-Dup by construction.
* Model identifiers in the configs (`qwen`, `gpt-5.6-luna`) are placeholders;
  set them to the exact checkpoints / API model ids.

## Citation

```bibtex
@inproceedings{memtrim2027,
  title     = {Understanding and Mitigating Inference-Time Overreliance Using Agentic Memory},
  author    = {Anonymous},
  booktitle = {Under review at ICLR},
  year      = {2027}
}
```
