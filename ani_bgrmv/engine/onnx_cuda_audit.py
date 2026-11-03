"""Strict-GPU audit of an ONNX Runtime CUDA profile.

``audit_cuda_profile`` inspects the kernels ONNX Runtime *executed* on the
first GPU run of a session. Neural-network layers must run on CUDA; only
tiny shape/metadata operations may run on CPU. Anything else fails closed,
so a model never silently runs on the CPU in GPU mode.
"""
from __future__ import annotations

import json

HQSAM_DECODER_PART = 'HQ-SAM decoder'

# ORT deliberately places graph-shape/control operations on the CPU even if
# the encoder/decoder's neural-network layers run on CUDA. These nodes are
# unavoidable with the HQ-SAM ONNX export used here. Do not confuse EP placement
# of a shape calculation with a full model fallback.
#
# The operator name alone does not reveal its purpose: Equal/Mul/Where can
# operate on tiny shape tensors or on image masks, so the actual tensor shapes
# in the *runtime execution profile* are checked.
CPU_SHAPE_OPS = frozenset({
    'Shape', 'Size', 'Gather', 'Slice', 'Unsqueeze', 'Squeeze', 'Concat',
    'Reshape', 'Cast', 'Constant', 'ConstantOfShape', 'Range', 'Expand',
    'Identity',
})
CPU_CONDITIONAL_SHAPE_OPS = frozenset({'Equal', 'Mul', 'Where'})
# The HQ-SAM2 exported decoder contains attention-scale calculations that
# ONNX Runtime may assign to CPU (e.g. Sqrt/Div of a scalar head dimension).
# This exception is for the HQ-SAM DECODER ONLY; no array-valued Sqrt/Div may run on CPU.
HQSAM_SCALAR_ATTENTION_OPS = frozenset({'Sqrt', 'Div'})
HQSAM_ATTENTION_NODE_PREFIX = '/mask_decoder/transformer/'
# A shape vector containing a few coordinates/dimensions is metadata; a full
# image/mask/feature-map tensor is NOT. Keep this conservative.
CPU_SHAPE_MAX_ELEMENTS = 256
CPU_SHAPE_MAX_RANK = 3
GPU_CORE_OPS = frozenset({
    'Conv', 'FusedConv', 'MatMul', 'Gemm', 'Attention', 'MultiHeadAttention',
    'FusedMatMul', 'QLinearConv', 'QLinearMatMul', 'MatMulInteger',
})


def _profile_tensor_shapes(raw) -> list[tuple[int, ...]] | None:
    """Read ORT's input_type_shape/output_type_shape, e.g. [{'int64':[2]}].

    Different ORT releases serialize this as a list of dtype-to-shape dicts
    or nested dicts; reject missing/unknown shapes instead of guessing.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    shapes = []

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                shapes.append(tuple(value))
            else:
                for item in value:
                    walk(item)

    walk(raw)
    return shapes or None


def _is_tiny_metadata_node(event: dict) -> tuple[bool, str]:
    """Only allow CPU Equal/Mul/Where on provably *tiny* tensors.

    Never blindly whitelist the operator name: a CPU Mul over a 256x256 mask
    would violate strict CUDA inference. A missing ORT shape is also a failure,
    not a reason to permit the operation.
    """
    args = event.get('args') or {}
    inputs = _profile_tensor_shapes(args.get('input_type_shape'))
    outputs = _profile_tensor_shapes(args.get('output_type_shape'))
    if not inputs or not outputs:
        return False, f'missing tensor shape metadata (input={inputs}, output={outputs})'
    for shape in inputs + outputs:
        if len(shape) > CPU_SHAPE_MAX_RANK:
            return False, f'non-metadata tensor rank={len(shape)}, shape={shape}'
        elements = 1
        for dim in shape:
            if dim < 0:
                return False, f'unknown tensor shape={shape}'
            elements *= dim
        if elements > CPU_SHAPE_MAX_ELEMENTS:
            return False, f'tensor too large (elements={elements}), shape={shape}'
    return True, f'input={inputs}, output={outputs}'


def _is_hqsam_scalar_attention_node(event: dict, part: str) -> tuple[bool, str]:
    """Permit HQ-SAM decoder attention scaling on CPU *only* for scalar tensors.

    Important: a node named Div or Sqrt is not automatically harmless. Require
    the exact HQ-SAM decoder family, expected attention path AND ORT runtime shape
    evidence for every operand/result. If shape metadata is missing, fail closed.
    """
    args = event.get('args') or {}
    op = str(args.get('op_name', ''))
    name = str(event.get('name', ''))
    if part != HQSAM_DECODER_PART or op not in HQSAM_SCALAR_ATTENTION_OPS:
        return False, 'not an HQ-SAM decoder scalar-attention operation'
    if not name.startswith(HQSAM_ATTENTION_NODE_PREFIX):
        return False, 'outside verified HQ-SAM transformer attention path'
    op_name = name.rsplit('/', 1)[-1]
    if not (op_name == op + '_kernel_time' or
            (op_name.startswith(op + '_') and op_name.endswith('_kernel_time'))):
        return False, 'profile node name does not match op_name'
    inputs = _profile_tensor_shapes(args.get('input_type_shape'))
    outputs = _profile_tensor_shapes(args.get('output_type_shape'))
    if not inputs or not outputs:
        return False, f'missing tensor shape metadata (input={inputs}, output={outputs})'
    for shape in inputs + outputs:
        if len(shape) > CPU_SHAPE_MAX_RANK or any(dim < 0 for dim in shape):
            return False, f'invalid scalar tensor shape={shape}'
        elements = 1
        for dim in shape:
            elements *= dim
        if elements != 1:
            return False, f'non-scalar tensor shape={shape}, elements={elements}'
    return True, f'proven scalar attention scaling: input={inputs}, output={outputs}'


def audit_cuda_profile(profile_events: list[dict], part: str) -> dict:
    """Audit *executed* ONNX kernels, not merely listed providers.

    CUDA must execute actual neural-network layers. CPU is permitted for
    metadata/shape operations only. Equal/Mul/Where qualify solely when ORT
    profiling confirms that all input/output tensors are tiny; otherwise the
    audit fails closed with evidence needed to debug/re-export the model.
    """
    events = [event for event in profile_events
              if event.get('cat') == 'Node'
              and str(event.get('name', '')).endswith('_kernel_time')]
    if not events:
        raise RuntimeError(
            f'{part}: ONNX Runtime không ghi lại node profiling; '
            'không thể xác minh chế độ GPU nghiêm ngặt.'
        )
    cpu = []
    cuda = []
    unknown = []
    allowed_cpu = []
    rejected = []
    for event in events:
        args = event.get('args') or {}
        provider = str(args.get('provider', ''))
        op = str(args.get('op_name', '<unknown>'))
        name = str(event.get('name', '?'))
        entry = (op, name)
        if provider == 'CUDAExecutionProvider':
            cuda.append(entry)
        elif provider == 'CPUExecutionProvider':
            cpu.append(entry)
            if op in CPU_SHAPE_OPS:
                allowed_cpu.append(entry)
            elif op in CPU_CONDITIONAL_SHAPE_OPS:
                tiny, evidence = _is_tiny_metadata_node(event)
                if tiny:
                    allowed_cpu.append(entry)
                else:
                    rejected.append((op, name, evidence))
            elif op in HQSAM_SCALAR_ATTENTION_OPS:
                scalar, evidence = _is_hqsam_scalar_attention_node(event, part)
                if scalar:
                    allowed_cpu.append(entry)
                else:
                    rejected.append((op, name, evidence))
            else:
                rejected.append((op, name, 'not a permitted shape/control operator'))
        else:
            unknown.append((provider, op, name))
    gpu_core = [(op, name) for op, name in cuda if op in GPU_CORE_OPS]
    if rejected or unknown or not gpu_core:
        bad_ops = sorted({op for op, _, _ in rejected})
        unknown_ep = sorted({p for p, _, _ in unknown})
        examples = '; '.join(
            f'{op}/{name}: {reason}' for op, name, reason in rejected[:5]
        ) or 'none'
        raise RuntimeError(
            f'{part}: GPU strict audit FAILED. '
            f'CUDA nodes={len(cuda)}, CUDA core layers={len(gpu_core)}, '
            f'CPU permitted auxiliary nodes={len(allowed_cpu)}; '
            f'CPU unsupported operations={bad_ops}; other providers={unknown_ep}. '
            f'CPU rejected node details: {examples}. '
            f'CPU Equal/Mul/Where chỉ được phép trên tensor shape/metadata nhỏ '
            f'(<= {CPU_SHAPE_MAX_ELEMENTS} phần tử, rank <= {CPU_SHAPE_MAX_RANK}) '
            'với profiling đủ dữ liệu. Không chạy model trên CPU thay thế.'
        )
    return {
        'part': part,
        'cuda_nodes': len(cuda),
        'cuda_core_layers': len(gpu_core),
        'cpu_shape_nodes': len(allowed_cpu),
        'cpu_shape_ops': sorted({op for op, _ in allowed_cpu}),
        'cpu_conditional_shape_nodes': sum(1 for op, _ in allowed_cpu
                                           if op in CPU_CONDITIONAL_SHAPE_OPS),
        'cpu_scalar_attention_nodes': sum(1 for op, _ in allowed_cpu
                                               if op in HQSAM_SCALAR_ATTENTION_OPS),
    }
