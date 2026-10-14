#!/usr/bin/env bash
set -euo pipefail

run_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "${run_dir}/kernel_tune.py" "$@"
