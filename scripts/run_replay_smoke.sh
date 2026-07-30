#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE=${1:-"$ROOT/examples/replay-smoke"}
OUTPUT=${2:-"${TMPDIR:-/tmp}/reprforge-replay-smoke"}

mkdir -p "$OUTPUT"

for policy in all-text all-image fixed-hybrid; do
  python -m reprforge.policy_replay \
    --items "$FIXTURE/items.jsonl" \
    --queries "$FIXTURE/queries.jsonl" \
    --scores "$FIXTURE/scores.jsonl" \
    --policy "$policy" \
    --target-metric recall_at_1 \
    --output "$OUTPUT/$policy.json"
done

python -m reprforge.policy_replay \
  --items "$FIXTURE/items.jsonl" \
  --queries "$FIXTURE/queries.jsonl" \
  --scores "$FIXTURE/scores.jsonl" \
  --policy exact-oracle \
  --index-budget-bytes 10 \
  --target-metric recall_at_1 \
  --output "$OUTPUT/exact-oracle.json"

echo "Replay smoke results: $OUTPUT"
