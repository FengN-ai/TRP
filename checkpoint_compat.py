"""Strict evaluation-only loading of ordinary and bitsandbytes 4-bit checkpoints.

Linear4bit saves quantization metadata alongside weight, but some bitsandbytes
versions do not consume it in nn.Module.load_state_dict. Restore it through
Params4bit.from_prequantized in temporary per-layer pre-hooks, then let PyTorch
perform its normal strict checks. Training/checkpoint-writing code is unchanged.

Reference: bitsandbytes.nn.Params4bit.from_prequantized and the Transformers
4-bit checkpoint loader:
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/quantizers/quantizer_bnb_4bit.py
"""

from collections.abc import Mapping
import inspect
import math

import torch


_QUANT_SUFFIXES = {
    "absmax", "quant_map", "nested_absmax", "nested_quant_map",
    "quant_state.bitsandbytes__nf4", "quant_state.bitsandbytes__fp4",
}


def load_quantized_state_dict(model, state_dict, strict=True):
    """Return (PyTorch incompatible-keys result, quantization restoration report).

    Only Linear4bit's recognized metadata is consumed, after successful restore.
    Other missing/unexpected keys and shape mismatches keep PyTorch's semantics.
    Broken or incomplete quantized weights always fail, even if strict=False.
    The caller's mapping/tensors are not edited and hooks are always removed.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a model state_dict mapping")
    # PEFT wrappers expose base_layer.weight through a property; only restore
    # the module that directly owns the parameter (and its state_dict keys).
    candidates = [module for module in model.modules()
                  if type(module._parameters.get("weight")).__name__ == "Params4bit"]
    restored = []
    handles = []
    if candidates:
        from bitsandbytes.nn import Linear4bit, Params4bit
        from bitsandbytes.utils import unpack_tensor_to_dict

        supports_module = "module" in inspect.signature(Params4bit.from_prequantized).parameters

        def restore(module, local_state, prefix, local_metadata, strict_flag,
                    missing_keys, unexpected_keys, error_msgs):
            weight_key = prefix + "weight"
            stat_prefix = weight_key + "."
            keys = [stat_prefix + suffix for suffix in _QUANT_SUFFIXES
                    if stat_prefix + suffix in local_state]
            packed = local_state.get(weight_key)
            if not keys:
                if packed is not None and (
                    packed.dtype == torch.uint8 or
                    (getattr(module.weight, "bnb_quantized", False) and
                     tuple(packed.shape) != (module.out_features, module.in_features))
                ):
                    raise RuntimeError(f"{weight_key}: packed 4-bit weight has no quantization metadata; refusing to reuse freshly initialized scales")
                return
            if packed is None:
                raise RuntimeError(f"{weight_key}: quantization metadata exists but weight is missing")
            stats = {key[len(stat_prefix):]: local_state[key] for key in keys}
            markers = [key for key in stats if key.startswith("quant_state.bitsandbytes__")]
            if len(markers) != 1:
                raise RuntimeError(f"{weight_key}: expected exactly one NF4/FP4 quant_state record")
            try:
                metadata = unpack_tensor_to_dict(stats[markers[0]])
                required = {"absmax", "quant_map", markers[0]}
                nested = any(key.startswith("nested_") for key in metadata)
                if nested:
                    required.update(("nested_absmax", "nested_quant_map"))
                missing = required - stats.keys()
                if missing:
                    raise ValueError(f"missing quantization components: {sorted(missing)}")
                if not nested and any(key.startswith("nested_") for key in stats):
                    raise ValueError("nested quantization tensors disagree with quant_state metadata")
                shape = tuple(metadata["shape"])
                expected_shape = (module.out_features, module.in_features)
                if shape != expected_shape:
                    raise ValueError(f"logical weight shape {shape} != layer shape {expected_shape}")
                expected_type = markers[0].rsplit("__", 1)[1]
                if metadata["quant_type"] != expected_type:
                    raise ValueError("quantization type disagrees with metadata key")
                if packed.numel() * packed.element_size() != (math.prod(shape) + 1) // 2:
                    raise ValueError("packed weight byte count does not match its logical 4-bit shape")
                blocksize = metadata["blocksize"]
                if blocksize <= 0 or stats["absmax"].numel() != math.ceil(math.prod(shape) / blocksize):
                    raise ValueError("absmax count does not match quantization blocks")
                # Avoid aliasing the CPU checkpoint tensor during CPU round-trip
                # tests. For CUDA this performs the single required H2D copy.
                data = packed.to(module.weight.device, copy=True)
                kwargs = {"module": module} if supports_module else {}
                weight = Params4bit.from_prequantized(
                    data=data, quantized_stats=dict(stats), requires_grad=False,
                    device=module.weight.device, **kwargs,
                )
            except Exception as exc:
                raise RuntimeError(f"Cannot restore {weight_key} and its quantization state: {exc}") from exc
            module.weight = weight
            module.quant_state = weight.quant_state
            # Native Module loading still checks/copies weight and bias. Only
            # metadata just consumed by from_prequantized is removed locally.
            local_state[weight_key] = weight.detach()
            for key in keys:
                local_state.pop(key)
            restored.append({"weight": weight_key, "quant_type": weight.quant_type,
                             "logical_shape": list(shape), "metadata_keys": len(keys)})

        unsupported = [module for module in candidates if not isinstance(module, Linear4bit)]
        if unsupported:
            raise TypeError(f"Unsupported 4-bit module for this evaluation loader: {type(unsupported[0]).__name__}")
        for module in candidates:
            handles.append(module.register_load_state_dict_pre_hook(restore))
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    finally:
        for handle in handles:
            handle.remove()
    return incompatible, {"restored_4bit_layers": len(restored),
                          "restored_quantization_state_keys": sum(item["metadata_keys"] for item in restored),
                          "layers": restored}


def load_evaluation_checkpoint(path, model, strict=True):
    """Load a repository {'model': state_dict, ...} checkpoint or bare state_dict.

    mmap avoids eagerly materializing the unused optimizer state in host RAM.
    Returns checkpoint, incompatible keys, and quantization restoration details.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping with 'model' or a bare state_dict")
    state = checkpoint.get("model", checkpoint)
    incompatible, restoration = load_quantized_state_dict(model, state, strict=strict)
    print(f"[Checkpoint] Restored {restoration['restored_4bit_layers']} 4-bit layers "
          f"({restoration['restored_quantization_state_keys']} quantization-state keys); strict={strict}", flush=True)
    return checkpoint, incompatible, restoration
