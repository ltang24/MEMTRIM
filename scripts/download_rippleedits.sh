#!/usr/bin/env bash
set -euo pipefail

COMMIT=54f3b88af4895a3aacb580ec63ce7ae857185040
BASE="https://raw.githubusercontent.com/edenbiran/RippleEdits/${COMMIT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/RippleEdits"

mkdir -p "${OUT}/data/benchmark"

curl -fsSL "${BASE}/LICENSE.txt" -o "${OUT}/LICENSE.txt"
for name in random popular recent; do
  curl -fsSL "${BASE}/data/benchmark/${name}.json" -o "${OUT}/data/benchmark/${name}.json"
done

cd "${OUT}"
sha256sum -c - <<'EOF'
c2b88d97cb5a0cddb17e9c2dc8d4207d3b77d878653ab3fe4999226ae454b14b  LICENSE.txt
f99e625e6b00225498d4843f4e3a688b391cdd98e6f9ee85608f72a1e28c2d40  data/benchmark/random.json
fe668f11871d3e1df8039be89124d5d39a443db0e9d1ded623d190f9e6a3296e  data/benchmark/popular.json
bc71807ea72280e2587bd1974fc79ae7e23aa24b1d5887bcdced40d448910547  data/benchmark/recent.json
EOF

echo "RippleEdits ${COMMIT} downloaded to ${OUT}"
