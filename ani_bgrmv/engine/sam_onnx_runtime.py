"""SAM ViT-B ONNX inference, matching Meta SamPredictor semantics.

No PyTorch, torchvision or segment_anything import at runtime.
Expected graphs come from tools/export_sam_vit_b_to_onnx.py:
  encoder(pixel_values[1,3,1024,1024]) -> image_embeddings[1,256,64,64]
  decoder(image_embeddings, point_coords, point_labels, mask_input,
          has_mask_input, orig_im_size) -> masks, iou_predictions, low_res_masks
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

import numpy as np
import onnxruntime as ort
from PIL import Image

SAM_SIZE = 1024
SAM_MEAN = np.asarray((123.675, 116.28, 103.53), dtype=np.float32).reshape(1, 1, 3)
SAM_STD = np.asarray((58.395, 57.12, 57.375), dtype=np.float32).reshape(1, 1, 3)


def sam_resize_shape(height: int, width: int, long_side: int = SAM_SIZE) -> tuple[int, int]:
    """Meta ResizeLongestSide.get_preprocess_shape, including +0.5 rounding."""
    scale = float(long_side) / max(height, width)
    return int(height * scale + 0.5), int(width * scale + 0.5)


def sam_preprocess(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Same PIL bilinear resizing, normalization and bottom/right zero pad as SAM.

    Note: padding must occur AFTER normalization (SAM.preprocess), not on raw RGB.
    """
    h, w = image_rgb.shape[:2]
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or h < 1 or w < 1:
        raise ValueError('SAM cần ảnh RGB H×W×3.')
    new_h, new_w = sam_resize_shape(h, w)
    resized = np.asarray(
        Image.fromarray(np.ascontiguousarray(image_rgb.astype(np.uint8)), 'RGB').resize(
            (new_w, new_h), resample=Image.Resampling.BILINEAR
        ), dtype=np.float32
    )
    normalized = (resized - SAM_MEAN) / SAM_STD
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = np.zeros((1, 3, SAM_SIZE, SAM_SIZE), dtype=np.float32)
    tensor[0, :, :new_h, :new_w] = chw
    return tensor, (new_h, new_w)


def transformed_coords(coords: np.ndarray, original_hw: tuple[int, int], resized_hw: tuple[int, int]):
    """Match ResizeLongestSide.apply_coords: scale x and y independently."""
    h, w = original_hw
    new_h, new_w = resized_hw
    output = np.asarray(coords, dtype=np.float32).copy()
    output[..., 0] *= float(new_w) / w
    output[..., 1] *= float(new_h) / h
    return output


def _validate_graph(session: ort.InferenceSession, expected_inputs: set[str], expected_outputs: set[str], label: str):
    names = {i.name for i in session.get_inputs()}
    outputs = {o.name for o in session.get_outputs()}
    missing_in, missing_out = expected_inputs - names, expected_outputs - outputs
    if missing_in or missing_out:
        raise RuntimeError(
            f'Graph SAM {label} không giống exporter AniBgRmv. '
            f'Thiếu input={sorted(missing_in)}, output={sorted(missing_out)}. '
            'Hãy export cả encoder và decoder từ cùng một checkpoint SAM.'
        )


# ORT deliberately places graph-shape/control operations on the CPU even if
# the encoder/decoder's neural-network layers run on CUDA. These nodes are
# unavoidable with the SAM ONNX export used here. Do not confuse EP placement
# of a shape calculation with a full model fallback.
#
# M42 assumed the operator name alone reveals its purpose. It does NOT:
# Equal/Mul/Where can operate either on tiny shape tensors or on image masks.
# M43 checks the actual tensor shapes in the *runtime execution profile*.
CPU_SHAPE_OPS = frozenset({
    'Shape', 'Size', 'Gather', 'Slice', 'Unsqueeze', 'Squeeze', 'Concat',
    'Reshape', 'Cast', 'Constant', 'ConstantOfShape', 'Range', 'Expand',
    'Identity',
})
CPU_CONDITIONAL_SHAPE_OPS = frozenset({'Equal', 'Mul', 'Where'})
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
            f'SAM {part}: ONNX Runtime không ghi lại node profiling; '
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
            f'SAM {part}: GPU strict audit FAILED. '
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
    }


class SamOnnxPredictor:
    """Drop-in narrow replacement for SamPredictor.set_image()/predict()."""

    def __init__(self, encoder_path: Path, decoder_path: Path, device: str = 'cpu'):
        encoder_path, decoder_path = Path(encoder_path), Path(decoder_path)
        for path in (encoder_path, decoder_path):
            if not path.is_file():
                raise FileNotFoundError(f'Không tìm thấy SAM ONNX: {path}')
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.device = device
        self.gpu = device.startswith('cuda:')
        self.audit_report: dict[str, dict] = {}
        self._audit_errors: dict[str, str] = {}
        if self.gpu:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('onnxruntime-gpu chưa có CUDAExecutionProvider; không fallback CPU.')
            idx = int(device.split(':', 1)[1])
            from ..app_config import APP_CONFIG
            from ..gpu_memory_options import cuda_provider_options
            self.providers = [(
                'CUDAExecutionProvider',
                cuda_provider_options(idx, settings=APP_CONFIG.get('compute', {})),
            )]
        elif device == 'cpu':
            self.providers = ['CPUExecutionProvider']
        else:
            raise ValueError(f'SAM chỉ hỗ trợ cpu hoặc cuda:N, nhận được: {device}')

        # M48: the encoder is only needed once to compute image_embeddings.
        # When a disk cache is available, it is not needed AT ALL.
        # Keep the small decoder alive across Guided strokes, and lazy-load
        # and immediately release the large ViT-B encoder on cache miss.
        self.encoder = None
        self.decoder = self._create_session(decoder_path, 'decoder')
        _validate_graph(
            self.decoder,
            {'image_embeddings', 'point_coords', 'point_labels', 'mask_input', 'has_mask_input', 'orig_im_size'},
            {'masks', 'iou_predictions', 'low_res_masks'}, 'decoder'
        )
        self.embedding: np.ndarray | None = None
        self.original_hw: tuple[int, int] | None = None
        self.resized_hw: tuple[int, int] | None = None

    def _create_session(self, path: Path, part: str):
        if self.gpu:
            opts = ort.SessionOptions()
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            # Shape-only CPU ops are allowed by the M43 per-node audit.
            # Disabling all CPU EP placement prevents SAM loading entirely.
            if part not in self.audit_report:
                opts.enable_profiling = True
                opts.profile_file_prefix = str(
                    Path(tempfile.gettempdir()) / f'anibgrmv_sam_{part}_{os.getpid()}'
                )
        else:
            from ..cpu_optimizations import cpu_session_options
            opts = cpu_session_options()
        session = ort.InferenceSession(str(path), sess_options=opts, providers=self.providers)
        chosen = session.get_providers()
        if self.gpu and (not chosen or chosen[0] != 'CUDAExecutionProvider'):
            raise RuntimeError(f'SAM {part} không nạp được CUDA trên {self.device}; không fallback CPU.')
        if not self.gpu and chosen != ['CPUExecutionProvider']:
            raise RuntimeError(f'SAM {part} không chạy đúng CPUExecutionProvider.')
        if self.gpu:
            session.disable_fallback()  # runtime failure must NOT rerun on CPU
        return session


    def _audit_first_gpu_run(self, part: str, session):
        if not self.gpu:
            return
        if part in self._audit_errors:
            raise RuntimeError(self._audit_errors[part])
        if part in self.audit_report:
            return
        file_name = session.end_profiling()
        if not file_name:
            reason = f'SAM {part}: profiling không tạo file; GPU chưa được xác minh.'
            self._audit_errors[part] = reason
            raise RuntimeError(reason)
        try:
            with Path(file_name).open('r', encoding='utf-8') as f:
                report = audit_cuda_profile(json.load(f), part)
            self.audit_report[part] = report
        except Exception as exc:
            self._audit_errors[part] = str(exc)
            raise
        finally:
            try:
                Path(file_name).unlink(missing_ok=True)
            except OSError:
                pass


    def set_cached_embedding(self, embedding, original_hw, resized_hw):
        """Restore a validated on-disk embedding; skip the expensive encoder."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (1, 256, 64, 64):
            raise ValueError('SAM embedding cache has incorrect shape')
        self.embedding = np.ascontiguousarray(embedding)
        self.original_hw = tuple(int(x) for x in original_hw)
        self.resized_hw = tuple(int(x) for x in resized_hw)

    def close(self):
        """Drop references to ORT sessions and the current image embedding."""
        self.embedding = None
        self.original_hw = None
        self.resized_hw = None
        self.encoder = None
        self.decoder = None

    def set_image(self, image_rgb: np.ndarray):
        input_tensor, resized_hw = sam_preprocess(image_rgb)
        # Encoder session is deliberately transient. Keeping it after one
        # image embedding causes large CUDA arenas to remain allocated for
        # the entire Edit session, despite later strokes only using decoder.
        self.encoder = self._create_session(self.encoder_path, 'encoder')
        try:
            _validate_graph(self.encoder, {'pixel_values'}, {'image_embeddings'}, 'encoder')
            embedding = self.encoder.run(['image_embeddings'], {'pixel_values': input_tensor})[0]
            self._audit_first_gpu_run('encoder', self.encoder)
            if tuple(embedding.shape) != (1, 256, 64, 64):
                raise RuntimeError(f'Embedding SAM không hợp lệ: {embedding.shape}, cần (1,256,64,64)')
            # A detached 4 MiB host buffer must not keep any ORT output arena alive.
            self.embedding = np.array(embedding, dtype=np.float32, order='C', copy=True)
            self.original_hw = tuple(int(x) for x in image_rgb.shape[:2])
            self.resized_hw = resized_hw
        finally:
            # The embedding is a detached numpy output, not a view of the ORT
            # session. Drop the encoder even when validation/inference fails.
            self.encoder = None

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        if self.embedding is None or self.original_hw is None or self.resized_hw is None:
            raise RuntimeError('Cần set_image() trước khi gọi predict() cho SAM ONNX.')
        coords = np.empty((0, 2), dtype=np.float32)
        labels = np.empty((0,), dtype=np.float32)
        if point_coords is not None:
            coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
            labels = np.asarray(point_labels, dtype=np.float32).reshape(-1)
            if len(labels) != len(coords):
                raise ValueError('SAM point_coords và point_labels khác chiều dài.')
        if box is not None:
            box = np.asarray(box, dtype=np.float32).reshape(2, 2)
            coords = np.concatenate([coords, box], axis=0)
            labels = np.concatenate([labels, np.asarray((2, 3), dtype=np.float32)], axis=0)
        if not len(coords):
            raise ValueError('SAM Guided cần point hoặc box prompt.')
        coords = transformed_coords(coords, self.original_hw, self.resized_hw)
        h, w = self.original_hw
        feeds = {
            'image_embeddings': self.embedding,
            'point_coords': np.ascontiguousarray(coords[np.newaxis], dtype=np.float32),
            'point_labels': np.ascontiguousarray(labels[np.newaxis], dtype=np.float32),
            'mask_input': np.zeros((1, 1, 256, 256), dtype=np.float32),
            'has_mask_input': np.zeros((1,), dtype=np.float32),
            'orig_im_size': np.asarray((h, w), dtype=np.float32),
        }
        masks_logits, scores, low_res = self.decoder.run(
            ['masks', 'iou_predictions', 'low_res_masks'], feeds
        )
        self._audit_first_gpu_run('decoder', self.decoder)
        masks_logits = np.asarray(masks_logits)
        scores = np.asarray(scores)
        low_res = np.asarray(low_res)
        if (masks_logits.ndim != 4 or masks_logits.shape[0] != 1
                or tuple(masks_logits.shape[-2:]) != (h, w)
                or scores.shape != (1, masks_logits.shape[1])
                or masks_logits.shape[1] != 4):
            raise RuntimeError(f'SAM decoder output không đúng định dạng (4 masks): '
                               f'{masks_logits.shape}, scores={scores.shape}; ảnh={(h,w)}')
        # SamPredictor.predict(multimask_output=True) returns candidates 1..3,
        # whereas official SamOnnxModel(return_single_mask=False) exports 0..3.
        chosen = slice(1, 4) if multimask_output else slice(0, 1)
        return (np.asarray(masks_logits[0, chosen] > 0.0, dtype=bool),
                np.asarray(scores[0, chosen], dtype=np.float32),
                np.asarray(low_res[0, chosen], dtype=np.float32))
