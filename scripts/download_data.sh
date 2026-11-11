#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fetch() {
  local base="$1" out="$2" path="$3"
  mkdir -p "$(dirname "${out}/${path}")"
  curl -fsSL "${base}/${path}" -o "${out}/${path}"
}

RIPPLE_COMMIT=54f3b88af4895a3aacb580ec63ce7ae857185040
RIPPLE_BASE="https://raw.githubusercontent.com/edenbiran/RippleEdits/${RIPPLE_COMMIT}"
RIPPLE_OUT="${ROOT}/RippleEdits"
fetch "${RIPPLE_BASE}" "${RIPPLE_OUT}" LICENSE.txt
for name in random popular recent; do
  fetch "${RIPPLE_BASE}" "${RIPPLE_OUT}" "data/benchmark/${name}.json"
done
(cd "${RIPPLE_OUT}" && sha256sum -c - <<'SUMS'
c2b88d97cb5a0cddb17e9c2dc8d4207d3b77d878653ab3fe4999226ae454b14b  LICENSE.txt
f99e625e6b00225498d4843f4e3a688b391cdd98e6f9ee85608f72a1e28c2d40  data/benchmark/random.json
fe668f11871d3e1df8039be89124d5d39a443db0e9d1ded623d190f9e6a3296e  data/benchmark/popular.json
bc71807ea72280e2587bd1974fc79ae7e23aa24b1d5887bcdced40d448910547  data/benchmark/recent.json
SUMS
)
echo "RippleEdits ${RIPPLE_COMMIT} downloaded to ${RIPPLE_OUT}"

COUNTERLOGIC_COMMIT=e35cd1125667f2c8a1596085750fecb1a4b0a0de
COUNTERLOGIC_BASE="https://raw.githubusercontent.com/Sukin272/CounterLogic/${COUNTERLOGIC_COMMIT}"
COUNTERLOGIC_OUT="${ROOT}/CounterLogic"
fetch "${COUNTERLOGIC_BASE}" "${COUNTERLOGIC_OUT}" Dataset/data_v3.json
(cd "${COUNTERLOGIC_OUT}" && sha256sum -c - <<'SUMS'
641547c3ada8740c62beb007ab4c49b5368c5121f86eb86e91304256e805f943  Dataset/data_v3.json
SUMS
)
echo "CounterLogic ${COUNTERLOGIC_COMMIT} downloaded to ${COUNTERLOGIC_OUT}"

CLADDER_COMMIT=3d2d1169b4b939a09048a6a75956c8972a93cc38
CLADDER_BASE="https://raw.githubusercontent.com/causalNLP/cladder/${CLADDER_COMMIT}"
CLADDER_OUT="${ROOT}/cladder"
fetch "${CLADDER_BASE}" "${CLADDER_OUT}" LICENSE
fetch "${CLADDER_BASE}" "${CLADDER_OUT}" data/cladder-v1.zip
(cd "${CLADDER_OUT}" && sha256sum -c - <<'SUMS'
5729188929e6873096ebe9623ae6ecb2164cb5d0acec10805cb4a463ff8fd05a  LICENSE
9fdae052b1ebe4ee6a19fdfb3e1eb88a381c7345df394cea524bcce97e2349b3  data/cladder-v1.zip
SUMS
)
echo "CLADDER ${CLADDER_COMMIT} downloaded to ${CLADDER_OUT}"
