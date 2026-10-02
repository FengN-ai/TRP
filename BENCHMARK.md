# Model size and inference cost

Run from the repository root in the model's CUDA environment. Provide the same architecture configuration used for the checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_trp_grasp.py infer \
  --config config_full.yaml \
  --checkpoint outputs/checkpoints/full/best_model.pth \
  --task both --seq-len 512 --batch-size 1 \
  --warmup 10 --iterations 100 \
  --output outputs/benchmarks/full
```

## Definitions

- **Total parameters:** logical parameter elements, including frozen and trainable parameters. Shared parameter objects are counted once. The counter reconstructs logical shapes for packed 4-bit weights. It includes registered parameters even if a submodule is not used in a particular forward pass; buffers and activation tensors are excluded.
- **Trainable parameters:** logical parameter elements selected by `requires_grad` after applying the current stage-freezing rules. Counts for both stages are reported. The final top-level count corresponds to Stage 2; it is not a reconstruction of historical training flags stored in a checkpoint.
- **Latency:** mean wall-clock duration of a single `VLAModel.forward` call with CUDA synchronization before and after timing. Data loading, preprocessing, and host-to-device input transfer are outside this interval. Autoregressive generation and robot execution are not measured. Inference uses `eval()`, inference mode, and BF16 autocasting.
- **Throughput:** processed samples per second during a separate repeated-forward timing interval. It need not be the exact reciprocal of the separately measured mean latency.
- **GPU memory:** PyTorch peak allocated and reserved memory are reported separately from sampled process/device memory. GiB uses `2**30` bytes; GB uses `10**9` bytes. State which field and unit you report.
- **FLOPs/MACs:** operator-count estimates when supported. Unsupported operations can make counting incomplete or unavailable. An unavailable result is not zero. MAC-equivalents follow the convention `1 MAC = 2 FLOPs`; quantized counts describe dense-equivalent arithmetic.

The default input is synthetic and controls input shape. `--seq-len` counts the prepared text sequence; visual prefix tokens are additional. The same prepared batch is repeated. `--input data` replays one dataset batch, including its reference assistant tokens, so it is also a forward benchmark rather than a text-generation or accuracy evaluation.

Report the GPU, quantization, compute dtype, batch size, image sizes, text length, warm-up count, repetition count, and timing scope with published measurements. Do not mix these forward timings with another method's complete generation or diffusion pipeline and describe them as the same measurement.

## Outputs

The output directory contains `report.json`, `summary.csv`, `load_resources.csv`, and `inference_resources.csv`. Load-time and inference-time resource samples are separate. Keep `checkpoint_compat.py` alongside the benchmark script; strict key checking is the default for evaluation.

Reports include configuration paths and environment information. They are generated experiment artifacts and are excluded from the source release by default.

## Training resource monitoring

```bash
CUDA_VISIBLE_DEVICES=0,1 python benchmark_trp_grasp.py monitor \
  --gpu-ids 0,1 --output outputs/benchmarks/training -- \
  torchrun --standalone --nproc_per_node=2 train.py --config config_full.yaml
```

`--gpu-ids` uses physical NVML indices or UUIDs. Monitoring observes the launched command and its workers, including initialization, validation, and checkpoint I/O. A saved checkpoint alone cannot provide past training time or energy use. Power/energy fields depend on available device telemetry.
