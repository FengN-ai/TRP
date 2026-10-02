#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
  full)
    CONFIG="$SCRIPT_DIR/config_full.yaml"
    ;;
  capd_only)
    CONFIG="$SCRIPT_DIR/config_capd_only.yaml"
    ;;
  vaf_only)
    CONFIG="$SCRIPT_DIR/config_vaf_only.yaml"
    ;;
  baseline)
    CONFIG="$SCRIPT_DIR/config_baseline.yaml"
    ;;
  separate_decoder)
    CONFIG="$SCRIPT_DIR/config_separate_decoder.yaml"
    ;;
  *)
    echo "Usage: bash run_labgrasp_train.sh [full|capd_only|vaf_only|baseline|separate_decoder]" >&2
    exit 2
    ;;
esac

cd "$SCRIPT_DIR"
python -m py_compile dataset.py model.py utils.py train.py \
  check_labgrasp_training.py
python check_labgrasp_training.py --config "$CONFIG"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NPROC="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29517}"

exec torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  "$SCRIPT_DIR/train.py" \
  --config "$CONFIG"
