#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${project_root}/python:${project_root}/examples${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${project_root}/examples/run/weft_runner.py" kernel "$@"
