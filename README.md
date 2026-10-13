# TRP-Grasp

Research code for task-conditioned grasping with VA-Fusion, intent-shared spatial decoding, and CAPD. The model combines DINOv2 and SigLIP visual features with a Qwen2 language backbone, a mask branch, and a planar grasp pose branch.


## Dataset

The **LabGrasp** dataset suite is publicly available on [Hugging Face](https://huggingface.co/datasets/xingyunyan/LabGrasp).

LabGrasp consists of three subsets for task-conditioned vision-language grasping in scientific laboratories.

| Subset | Archive size | Download |
|---|---:|---|
| LabGrasp-Basic | 49.8 GB | [Download](https://huggingface.co/datasets/xingyunyan/LabGrasp/resolve/main/LabGrasp-Basic.zip?download=true) |
| LabGrasp-Deploy-Syn | 1.32 GB | [Download](https://huggingface.co/datasets/xingyunyan/LabGrasp/resolve/main/LabGrasp-Deploy-Syn.zip?download=true) |
| LabGrasp-Deploy-Real | 6.87 GB | [Download](https://huggingface.co/datasets/xingyunyan/LabGrasp/resolve/main/LabGrasp-Deploy-Real.zip?download=true) |

### Download

Download the full dataset from the [LabGrasp Hugging Face repository](https://huggingface.co/datasets/xingyunyan/LabGrasp), or use the Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download xingyunyan/LabGrasp --repo-type dataset --include "*.zip" --local-dir ./downloads
```

To download a single subset, for example LabGrasp-Deploy-Real:

```bash
hf download xingyunyan/LabGrasp LabGrasp-Deploy-Real.zip --repo-type dataset --local-dir ./downloads
```

After downloading, extract the required archive and organize the data according to the paths specified in the training configurations. See [TRAINING.md](TRAINING.md) for details.
  

LabGrasp contains three subsets:

- **LabGrasp-Basic**
- **LabGrasp-Deploy-Syn**
- **LabGrasp-Deploy-Real**

After downloading, place the required dataset under the `data/` directory following the structure described below.

## Repository layout

| File | Purpose |
|---|---|
| `train.py` | Two-stage training and validation |
| `model.py` | Multimodal model, VA-Fusion, spatial decoders, and CAPD |
| `dataset.py` | Heterogeneous AG, PR, and VQA data loading |
| `utils.py` | Losses, metrics, optimizer, scheduler, and checkpoint utilities |
| `config_full.yaml` | Full-model training configuration |
| `config_*_only.yaml`, `config_baseline.yaml` | Module ablations |
| `config_separate_decoder.yaml` | Separate-decoder configuration; see the setting below |
| `run_labgrasp_train.sh` | Dataset preflight and training launcher |
| `check_labgrasp_training.py` | Annotation and dataset validation |
| `run_deep_ablations.py` | Independent training, evaluation, and aggregation of mechanism ablations |
| `benchmark_trp_grasp.py` | Forward-pass and resource benchmarks |
| `checkpoint_compat.py` | Strict evaluation loading of 4-bit checkpoints |

See [training and data setup](TRAINING.md), [mechanism ablations](DEEP_ABLATIONS.md), and [benchmark definitions](BENCHMARK.md).

## Environment

Use Linux, Python 3.10, and a CUDA-enabled PyTorch build compatible with your GPU. Install PyTorch first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The recorded inference environment used Python 3.10.19, PyTorch `2.11.0.dev20260104+cu128`, Transformers `5.3.0`, PEFT `0.18.1`, and bitsandbytes `0.49.2`. These are recorded benchmark versions, not a complete training-environment lockfile. CUDA, driver, and PyTorch compatibility must match the target hardware. The training launcher uses CUDA/NCCL; CPU training is not supported by the supplied launcher.

## Local resources

Run the documented commands from the repository root. All relative paths in YAML files resolve against the working directory, **not** the YAML file's directory. Place resources in the following layout, or edit the corresponding YAML path fields:

```text
TRP-Grasp/
  data/
    LabGrasp-Deploy-Real/
      annotations/
        train.jsonl
        val.jsonl
        test.jsonl
      ... image and mask files referenced by the annotations ...
  pretrained/
    qwen2-7b/
    dinov2-large/
    siglip/
  outputs/
    checkpoints/
    logs/
```

The three pretrained directories must contain local Hugging Face model snapshots; the language-model directory must also contain its tokenizer files. The supplied configuration expects Qwen2-7B (hidden size 3584), DINOv2-L (1024), and SigLIP-SO400M (1152). DINOv2 and SigLIP input resolutions are 336 and 384, respectively. Folder names are local placeholders, not download IDs. Use the pretrained variants corresponding to your experiment; weights and datasets are not bundled here.

Edit `data_root`, `train_jsonl`, `val_jsonl`, `test_jsonl`, `llm_name_or_path`, `dinov2_name`, and `siglip_name` if your resources are elsewhere. Use literal relative or absolute paths; YAML values do not expand shell variables or `~`. Image and mask paths inside the annotations are resolved relative to `data_root`.

## Train and evaluate

```bash
python check_labgrasp_training.py --config config_full.yaml
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash run_labgrasp_train.sh full
```

Stage 1 trains the introduced modules with the pretrained backbones frozen. Stage 2 additionally trains the Qwen2 LoRA parameters. The supplied full configuration uses 3 and 6 epochs, respectively, a per-GPU batch size of 1, and 16 gradient accumulation steps. Validation selects `best_model.pth`; `last_model.pth` stores the state for resuming training.

Evaluate a trained checkpoint on the configured test split:

```bash
CUDA_VISIBLE_DEVICES=0 python run_deep_ablations.py eval-one \
  --config config_full.yaml \
  --checkpoint-path outputs/checkpoints/full/best_model.pth \
  --output outputs/evaluation/full/test_metrics.json
```

For SSH sessions, use the background commands in [TRAINING.md](TRAINING.md).

## Ablation configurations

| Configuration | CAPD | VA-Fusion | Spatial decoder |
|---|---|---|---|
| `config_full.yaml` | Enabled | Enabled | Shared |
| `config_capd_only.yaml` | Enabled | Disabled | Shared |
| `config_vaf_only.yaml` | Disabled | Enabled | Shared |
| `config_baseline.yaml` | Disabled | Disabled | Shared |
| `config_separate_decoder.yaml` | Disabled | Disabled | Separate |

The separate-decoder file preserves the supplied experiment's `use_capd: false` and `use_vaf: false` settings. It isolates decoder sharing relative to `config_baseline.yaml`, not relative to the full model. Its checkpoint and log directories are distinct. To study decoder sharing with both mechanisms enabled, create an additional configuration based on `config_full.yaml`, set `use_shared_decoder: false`, and give that experiment its own output directories.

All supplied configurations exclude `pipette` for the eight-class setting. Keep data filtering, seeds, training schedules, and evaluation protocols consistent within each comparison.

## Resource benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_trp_grasp.py infer \
  --config config_full.yaml \
  --checkpoint outputs/checkpoints/full/best_model.pth \
  --task both --seq-len 512 --batch-size 1 \
  --warmup 10 --iterations 100 \
  --output outputs/benchmarks/full
```

This measures `VLAModel.forward`, not autoregressive text generation or robot execution. [BENCHMARK.md](BENCHMARK.md) defines parameter counts, timing boundaries, and memory units.

## Publication files

Datasets, model weights, checkpoints, experiment logs, benchmark reports, caches, credentials, and local archives are excluded by `.gitignore`. The repository contains source code and portable example configurations. Review generated reports separately before sharing them: they record runtime paths and environment metadata.
