# Training and data setup

Run all commands from the repository root after configuring the local resources described in [README.md](README.md). Use a separate configuration and output directory for each experiment.

## Annotation format

`train_jsonl`, `val_jsonl`, and `test_jsonl` point to distinct JSONL files. Each record uses `id`, `split`, `data_type`, `image`, `conversations`, `supervision`, and `ground_truth`. Conversation turns use `from` (`human` or `assistant`) and `value`.

| `data_type` | Supervision | Structured target |
|---|---|---|
| `Affordance_Grounding` | `mask: true`, `pose: false`, `language: true` | `ground_truth.mask_path`; assistant target contains `<GRASP_MASK>` |
| `Pose_Regression` | `mask: false`, `pose: true`, `language: true` | `ground_truth.pose` with `x`, `y`, `angle`, `width`; assistant target contains `<GRASP_POSE>` |
| `VQA` | `mask: false`, `pose: false`, `language: true` | Language target without either spatial token |

Pose centers and widths are expressed in the original image's pixel coordinates; angles are in degrees. Masks are binary images. `image` and `mask_path` are relative to `data_root`. Use `ground_truth.object_class` for class filtering. Each PR record must have a corresponding AG record in the same split, linked by `pair_id` or `related_pair_id`. The preflight checker verifies required resources, supervision fields, pairing, unique IDs, and image-basename overlap across splits.

```bash
python check_labgrasp_training.py --config config_full.yaml
```

## Launch

The launcher accepts `full`, `capd_only`, `vaf_only`, `baseline`, and `separate_decoder`:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash run_labgrasp_train.sh full
```

For a single GPU, set `CUDA_VISIBLE_DEVICES=0` and `NPROC_PER_NODE=1`. Keep the GPU count, per-GPU batch size, and accumulation settings identical across compared experiments.

For custom configurations, launch through `torchrun`, including for one GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --config config_full.yaml
```

## Run in the background

The following command writes output to a log file without streaming it to the SSH terminal:

```bash
mkdir -p outputs/console
RUN_LOG="outputs/console/full_$(date +%Y%m%d_%H%M%S).log"
nohup env CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=29517 \
  bash run_labgrasp_train.sh full \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > outputs/console/full.pid
echo "$RUN_LOG" > outputs/console/full_log_path.txt
```

Inspect recent progress when needed:

```bash
tail -n 30 "$(cat outputs/console/full_log_path.txt)"
```

`nohup` protects against an SSH hangup. It does not restart jobs after a GPU error, process failure, or server reboot.

## Checkpoints and resuming

The full-model configuration saves to `outputs/checkpoints/full/`. Validation cIoU selects `best_model.pth`; `last_model.pth` is saved at epoch boundaries. Resume the same experiment with its own checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --config config_full.yaml \
  --resume outputs/checkpoints/full/last_model.pth
```

Do not run two jobs writing to the same checkpoint directory. Resume uses the most recently saved training state; work after that save may need to be repeated. Training and evaluation retain their existing checkpoint-loading behavior.

## Test evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run_deep_ablations.py eval-one \
  --config config_full.yaml \
  --checkpoint-path outputs/checkpoints/full/best_model.pth \
  --output outputs/evaluation/full/test_metrics.json
```

`eval-one` is a single-GPU command and covers the entire configured test split. It reuses `train.evaluate()` and the existing metric definitions. The test split is not used to select the checkpoint. Keep `checkpoint_compat.py` with the repository for strict restoration of 4-bit quantization state.
