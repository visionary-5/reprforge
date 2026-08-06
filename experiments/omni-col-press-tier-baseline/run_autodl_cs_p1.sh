#!/usr/bin/env bash
set -euo pipefail

export AUTODL_DOMAIN=computer-science
export AUTODL_MAX_CORPUS_ROWS=1360
export AUTODL_MAX_QUERY_ROWS=215
export AUTODL_OUTPUT_ROOT="${AUTODL_OUTPUT_ROOT:-${AUTODL_DATA_ROOT:-/root/autodl-tmp}/outputs/omni-cs-p1-full-hpool-agc-cascade-12bfcb2-rtx5090}"
export OMNI_PROFILE_BATCHES="${OMNI_PROFILE_BATCHES:-1}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_autodl_domain_p1.sh"
