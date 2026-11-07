#!/usr/bin/env bash
# Full pipeline. Start your vLLM server (Qwen) and Letta server (MemGPT) first,
# and export OPENAI_API_KEY for the GPT runs. Each step resumes if interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/download_data.sh
python -m memtrim.data.build --dataset all                 # App. C -> data/processed/*.jsonl
python -m unittest discover -s tests                       # sanity

for cfg in main_qwen main_gpt; do                          # Table 1 (no_memory/memory rows), Tables 2-4
  python scripts/run.py --config configs/$cfg.yaml
  python -m memtrim.report results/$cfg
  python -m memtrim.experiments.rq2 results/$cfg           # RQ2: Figure 2 (RippleEdits), Figure 8 (CounterLogic)
done
python -m memtrim.experiments.rq3 --config configs/rq3.yaml           # RQ3: Figure 3, Figure 9
python scripts/run.py --config configs/ablation_qwen.yaml && \
  python -m memtrim.report results/ablation_qwen --ablation            # Figure 5
python -m memtrim.experiments.attack --config configs/attack.yaml     # Figure 6
python scripts/run.py --config configs/runtime_qwen.yaml && \
  python -m memtrim.report results/runtime_qwen --runtime              # Figure 7
