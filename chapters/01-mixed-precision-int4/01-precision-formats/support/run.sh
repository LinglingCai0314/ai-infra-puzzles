#!/usr/bin/env bash
set -euo pipefail

lesson_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CH1_OUTPUT_DIR="${CH1_OUTPUT_DIR:-$lesson_dir/outputs/raw}"
export CH1_REPORT_DIR="${CH1_REPORT_DIR:-$lesson_dir/outputs/reports}"

mkdir -p "$CH1_OUTPUT_DIR" "$CH1_REPORT_DIR"

python "$lesson_dir/support/benchmark.py" --mode bf16
python "$lesson_dir/support/benchmark.py" --mode int4
python "$lesson_dir/support/summarize.py"

printf 'Results: %s\n' "$CH1_REPORT_DIR/comparison.md"
