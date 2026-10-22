"""GPU ONNX Runtime memory controls for AniBgRmv M48.

This module does not import ONNX Runtime, PyTorch, or allocate GPU memory.
It only returns documented CUDAExecutionProvider option values.

These are performance/memory heuristics, NOT a total RAM or VRAM cap.
"""
from __future__ import annotations


def cuda_provider_options(device_id: int, *, settings: dict | None = None) -> dict[str, str]:
    """Build CUDA EP options without changing which device runs the model.

    Defaults prioritize predictable/low peak CUDA workspace over maximum
    convolution autotuning speed. Only use gpu_mem_limit when explicitly set:
    it limits the ORT arena, not total GPU memory, and an arbitrary 4 GB
    cap can break SAM inference with out-of-memory errors.
    """
    settings = settings or {}
    device_id = int(device_id)
    if device_id < 0:
        raise ValueError('CUDA device_id must be >= 0')

    out = {'device_id': str(device_id)}
    if not settings.get('gpu_low_memory_mode', True):
        return out  # exact legacy provider defaults; useful for A/B comparison

    # kSameAsRequested avoids arena growth in powers of two.
    out.update({
        'arena_extend_strategy': 'kSameAsRequested',
        'cudnn_conv_algo_search': 'HEURISTIC',
        'cudnn_conv_use_max_workspace': '0',
        'do_copy_in_default_stream': '1',
    })
    # Optional, never hard-coded for a GTX 1650's 4 GB VRAM.
    limit = settings.get('gpu_arena_limit_mib')
    if limit not in (None, '', 0, '0'):
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError('compute.gpu_arena_limit_mib must be an integer') from exc
        if not 128 <= limit <= 65536:
            raise ValueError('compute.gpu_arena_limit_mib must be between 128 and 65536')
        out['gpu_mem_limit'] = str(limit * 1024 * 1024)
    return out
