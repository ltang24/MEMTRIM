#!/usr/bin/env bash
# Fetch the three source benchmarks at the commits this code was written against.
set -euo pipefail
mkdir -p data/raw && cd data/raw
fetch() { # name url commit
  [ -d "$1/.git" ] || git clone "$2" "$1"
  git -C "$1" fetch -q origin && git -C "$1" checkout -q "$3"
  echo "$1 @ $(git -C "$1" rev-parse --short HEAD)"
}
fetch RippleEdits  https://github.com/edenbiran/RippleEdits   54f3b88af4895a3aacb580ec63ce7ae857185040
fetch cladder      https://github.com/causalNLP/cladder       3d2d1169b4b939a09048a6a75956c8972a93cc38
fetch CounterLogic https://github.com/Sukin272/CounterLogic   e35cd1125667f2c8a1596085750fecb1a4b0a0de
