# VA-Fusion and CAPD mechanism ablations

Run from the repository root. Each variant starts independently from the configured pretrained backbones, retaining CAPD, VA-Fusion, and the shared decoder. It does not initialize from a trained full-model checkpoint.

| Variant | Change relative to `full` |
|---|---|
| `full` | Full-model reference |
| `vaf_fixed_gate` | Fix the fusion gate to 0.5; retain viewpoint auxiliary supervision |
| `vaf_no_gate_supervision` | Retain the learned gate; set `lambda_vp: 0.0` |
| `capd_uniform_weights` | Replace confidence weights with a uniform mean within the same 5x5 window |
| `capd_no_progressive` | Fix the training sampling coefficient to 0.9; retain the consistency-loss schedule |
| `capd_point_sampling` | Replace the 5x5 window with single-point bilinear sampling |
| `capd_no_consistency` | Set `lambda_consist: 0.0` |

For `capd_no_progressive`, the training sampling center is `0.1 * gt_center + 0.9 * pred_center.detach()`. Inference uses the predicted center only. The original logged `capd_alpha` describes the consistency-loss schedule; it does not override the fixed sampling coefficient. This variant removes progressive centroid annealing, not the local sampling operation.

`vaf_fixed_gate` retains the auxiliary gate branch, so its viewpoint accuracy is not a measure of a learned fusion gate. `capd_point_sampling` removes local aggregation; its single point has relative weight 1. `capd_uniform_weights` isolates confidence weighting while preserving the local window.

## Prepare without training

```bash
python run_deep_ablations.py prepare \
  --base-config config_full.yaml --root outputs/deep_ablation_core \
  --seeds 42 \
  --variants full vaf_fixed_gate capd_uniform_weights capd_no_progressive
```

## Train and evaluate the core variants

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_deep_ablations.py run \
  --base-config config_full.yaml --root outputs/deep_ablation_core \
  --nproc 2 --seeds 42 \
  --variants full vaf_fixed_gate capd_uniform_weights capd_no_progressive
```

This launches four complete two-stage training runs sequentially. For three seeds, specify `--seeds 42 43 44`. Omitting `--variants` selects all seven variants, which means 21 training runs for three seeds. Keep GPU count and training settings consistent across a comparison.

To run the core variants in the background without streaming logs to SSH:

```bash
mkdir -p outputs/console
RUN_LOG="outputs/console/deep_core_$(date +%Y%m%d_%H%M%S).log"
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  python run_deep_ablations.py run \
  --base-config config_full.yaml --root outputs/deep_ablation_core \
  --nproc 2 --seeds 42 \
  --variants full vaf_fixed_gate capd_uniform_weights capd_no_progressive \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > outputs/console/deep_core.pid
echo "$RUN_LOG" > outputs/console/deep_core_log_path.txt
```

## Results and resuming

Each `<root>/<variant>/seed_<seed>/` contains its generated config, checkpoints, logs, status, and test metrics. The root contains `manifest.json`, `results_per_seed.csv`, and `results_summary.csv`. Aggregation reports means and sample standard deviations; a single seed has no standard deviation. Missing metrics remain empty.

Validation selects `best_model.pth`. Testing covers the entire configured test split through the existing `train.evaluate()` implementation. Use `--checkpoint last_model.pth` only when deliberately evaluating the last checkpoint; those results receive a separate suffix.

Add `--resume` to the same `run` command to continue unfinished runs from their own `last_model.pth`. Completed runs are skipped. Add `--no-evaluate` to train first and evaluate later:

```bash
CUDA_VISIBLE_DEVICES=0 python run_deep_ablations.py evaluate \
  --root outputs/deep_ablation_core
python run_deep_ablations.py summarize --root outputs/deep_ablation_core
```

The manifest checks configuration and source hashes. Use a new experiment root for this renamed publication copy; existing manifests from differently named source files will not match. Keep an existing experiment with the source revision and configuration that created it. The publication copy does not relax these checks.
