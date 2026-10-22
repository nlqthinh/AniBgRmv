"""Tune ONNX Runtime for small CPUs; does not alter input sizes or model weights."""
import os


def cpu_session_options():
    import onnxruntime as ort
    from .app_config import APP_CONFIG
    settings = APP_CONFIG.get('compute', {})
    requested = settings.get('cpu_threads', 2)
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = 2
    cpu_count = max(1, os.cpu_count() or 1)
    threads = min(cpu_count, max(1, min(requested, 8)))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Avoid reserving large CPU arenas/memory patterns for full-resolution
    # 768px inference on weak 2-core computers. May trade some throughput for
    # lower *retained* private memory; numerical results are unchanged.
    if bool(settings.get('low_memory_mode', True)):
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
    return opts
