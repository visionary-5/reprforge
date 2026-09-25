#!/usr/bin/env bash
# Run from an artifact directory containing code/ and config.json. Set PYTHON to the appropriate environment.
# The optional Qwen3 overlays remain user-provided; see docs/submission-coverage.md.
set -euo pipefail
task=$1
label=$2
shift 2
python=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$task" == qwen3 ]]; then
    export PYTHONPATH="$PWD/qwen3-extra:$PWD/qwen3-overlay-5.9:$PYTHONPATH"
fi
if [[ "$task" == quality ]]; then
    command=("$python" code/experiments/final_pass/quality_timing.py --config config.json --output "$label" "$@")
else
    command=("$python" code/experiments/final_pass/manual_interface.py --architecture "$task" --config config.json --output "$label" "$@")
fi
printf '%q ' "${command[@]}" > "$label.command.txt"
printf '\nCUDA_VISIBLE_DEVICES=%q\nPYTHONPATH=%q\nOMP_NUM_THREADS=%q\n' "$CUDA_VISIBLE_DEVICES" "$PYTHONPATH" "$OMP_NUM_THREADS" >> "$label.command.txt"
nvidia-smi > "$label.gpu-before.txt"
"$python" -m pip freeze > "$label.packages.txt"
set +e
"${command[@]}" > "$label.log" 2>&1
status=$?
printf '{"exit_code": %s}\n' "$status" > "$label.exit.json"
nvidia-smi > "$label.gpu-after.txt"
exit "$status"
