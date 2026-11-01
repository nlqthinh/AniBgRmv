"""HQ3 ONNX image predictor: point + box-prompt HQ-SAM2 Large, no PyTorch at runtime.

The exporter combines the HQ-SAM2 base checkpoint + HQ3 decoder state into
TWO compatible ONNX graphs. This runtime intentionally preserves the existing
Guided set_image/predict/predict_crop/close contract and HQ3 single-mask path.
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

SAM_SIZE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
ENCODER_OUTPUTS = ('high_res_feats_0', 'high_res_feats_1', 'image_embed')
DECODER_INPUTS = (*ENCODER_OUTPUTS, 'point_coords', 'point_labels')
DECODER_OUTPUTS = ('masks', 'iou_predictions')


def preprocess_hq3(image_rgb: np.ndarray) -> np.ndarray:
    """Match SAM2 1024x1024 RGB resize and ImageNet normalization."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('HQ3 ONNX cần ảnh RGB uint8 (H,W,3).')
    resized = np.asarray(Image.fromarray(np.ascontiguousarray(image), 'RGB').resize(
        (SAM_SIZE, SAM_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return np.ascontiguousarray(((resized - _MEAN) / _STD).transpose(2, 0, 1)[None])


def transform_hq3_points(coords: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    h, w = original_hw
    xy = np.asarray(coords, dtype=np.float32).reshape(-1, 2).copy()
    xy[:, 0] = xy[:, 0] / float(w) * SAM_SIZE
    xy[:, 1] = xy[:, 1] / float(h) * SAM_SIZE
    return xy


def check_hq3_onnx_contract(encoder_path: Path, decoder_path: Path) -> dict:
    """Reject mixed, unvalidated graph pairs before any CUDA allocation."""
    enc, dec = Path(encoder_path), Path(decoder_path)
    for p in (enc, dec):
        if not p.is_file():
            raise FileNotFoundError(f'Chưa có HQ3 ONNX: {p}. Chạy bat/prepare_hq3_onnx.bat.')
    manifest_path = enc.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        raise RuntimeError('Thiếu HQ3 ONNX manifest đã kiểm chứng: ' + str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('format') != 'anibgrmv-hq3-sam2-point-onnx-v1'
            or manifest.get('verified') is not True
            or manifest.get('encoder_filename') != enc.name
            or manifest.get('decoder_filename') != dec.name):
        raise RuntimeError('HQ3 ONNX không có manifest xác nhận export thành công; không dùng nhầm weights.')
    if enc.stat().st_size != manifest.get('encoder_size') or dec.stat().st_size != manifest.get('decoder_size'):
        raise RuntimeError('File HQ3 ONNX bị thay đổi sau export; cần export lại cùng checkpoint.')
    # A manifest is only meaningful if graphs, AND their external tensor data,
    # are the same exact export. This runs once when the predictor is loaded.
    def digest(path):
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                sha.update(chunk)
        return sha.hexdigest()

    if (digest(enc) != manifest.get('encoder_sha256') or
            digest(dec) != manifest.get('decoder_sha256')):
        raise RuntimeError('HQ3 ONNX SHA256 khác manifest; không nạp graph bị thay đổi.')
    for filename, info in manifest.get('sidecars', {}).items():
        if Path(filename).name != filename:
            raise RuntimeError('HQ3 ONNX sidecar có tên không an toàn.')
        part = enc.parent / filename
        if (not part.is_file() or part.stat().st_size != info['size'] or
                digest(part) != info['sha256']):
            raise RuntimeError('HQ3 ONNX thiếu/khác tensor sidecar: ' + filename)
    return manifest


class Hq3OnnxPredictor:
    """Small point/box API compatible with HQ3 PyTorch Guided backend."""

    def __init__(self, encoder_path: Path, decoder_path: Path, device: str):
        self.encoder_path, self.decoder_path = Path(encoder_path), Path(decoder_path)
        self.manifest = check_hq3_onnx_contract(self.encoder_path, self.decoder_path)
        self.device = device
        self.gpu = device.startswith('cuda:')
        if self.gpu:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('HQ3 ONNX cần CUDAExecutionProvider; không fallback CPU.')
            from ..app_config import APP_CONFIG
            from ..gpu_memory_options import cuda_provider_options
            idx = int(device.split(':', 1)[1])
            self.providers = [('CUDAExecutionProvider', cuda_provider_options(
                idx, settings=APP_CONFIG.get('compute', {})))]
        elif device == 'cpu':
            self.providers = ['CPUExecutionProvider']
        else:
            raise ValueError(f'Thiết bị HQ3 ONNX không hợp lệ: {device}')
        self._features = None
        self._original_hw = None
        self._audited: set[str] = set()
        # Hiera Large ONNX session construction/graph optimization is expensive.
        # A new session for EVERY crop caused long stalls during Guided strokes.
        # Keep one lazy encoder session until leaving Edit (close()).
        from ..app_config import APP_CONFIG
        opts = APP_CONFIG.get('sam', {})
        self._reuse_encoder_session = bool(opts.get('hq3_onnx_reuse_encoder_session', True))
        self._shrink_encoder_arena = bool(opts.get('hq3_onnx_encoder_shrink_gpu_arena_after_run', True))
        self._timing = bool(opts.get('hq3_onnx_timing', True))
        self._cpu_threads = max(1, min(4, int(opts.get('hq3_onnx_cpu_threads', 2))))
        self.encoder = None  # lazy to avoid peak GPU allocation on entering Edit
        self.decoder = self._session(self.decoder_path, 'decoder')
        self._validate_io(self.decoder, set(DECODER_INPUTS), set(DECODER_OUTPUTS), 'decoder')

    def _session(self, path: Path, part: str):
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        # Small CPU helper ops should not saturate the GUI machine with an
        # oversized ORT threadpool while QThread is running Guided.
        opts.intra_op_num_threads = getattr(self, '_cpu_threads', 2)
        opts.inter_op_num_threads = 1
        if self.gpu and part not in self._audited:
            opts.enable_profiling = True
        session = ort.InferenceSession(str(path), sess_options=opts, providers=self.providers)
        if self.gpu:
            if not session.get_providers() or session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError(f'HQ3 ONNX {part} không được nạp trên CUDA.')
            session.disable_fallback()
        return session

    @staticmethod
    def _validate_io(session, required_input: set[str], required_output: set[str], part: str):
        ins = {x.name for x in session.get_inputs()}
        outs = {x.name for x in session.get_outputs()}
        if not required_input.issubset(ins) or not required_output.issubset(outs):
            raise RuntimeError(f'HQ3 ONNX {part} sai graph: inputs={ins}; outputs={outs}.')

    def _audit(self, session, part):
        if not self.gpu or part in self._audited:
            return
        from .sam_onnx_runtime import audit_cuda_profile
        path = session.end_profiling()
        try:
            events = json.loads(Path(path).read_text(encoding='utf-8'))
            audit_cuda_profile(events, 'HQ3 ' + part)
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        self._audited.add(part)

    def _log_timing(self, event: str, elapsed: float) -> None:
        if getattr(self, '_timing', False):
            print(f'[AniBgRmv HQ3 ONNX] {event}_ms={elapsed * 1000:.1f}', flush=True)

    def _get_encoder(self):
        """Reuse one validated Hiera ONNX session for full images and crops.

        First session creation stays on the Guided worker thread, never on Qt's
        GUI thread. No extra model or package is loaded/downloaded.
        """
        encoder = getattr(self, 'encoder', None)
        if encoder is not None:
            return encoder
        t0 = time.perf_counter()
        encoder = self._session(self.encoder_path, 'encoder')
        try:
            self._validate_io(encoder, {'image'}, set(ENCODER_OUTPUTS), 'encoder')
        except Exception:
            del encoder
            raise
        self._log_timing('encoder_session_init', time.perf_counter() - t0)
        if getattr(self, '_reuse_encoder_session', True):
            self.encoder = encoder
        return encoder

    def _encoder_run_options(self):
        """Return only idle CUDA workspace, never unload cached model weights.

        Session reuse avoids reparsing/optimizing the Hiera graph. Arena shrink
        is optional because it can trade a little inference speed for 4GB VRAM.
        """
        if not self.gpu or not getattr(self, '_shrink_encoder_arena', True):
            return None
        opts = ort.RunOptions()
        opts.add_run_config_entry(
            'memory.enable_memory_arena_shrinkage',
            'gpu:' + self.device.split(':', 1)[1],
        )
        return opts

    def _encode(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        encoder = self._get_encoder()
        t0 = time.perf_counter()
        try:
            feed = {'image': preprocess_hq3(image)}
            options = self._encoder_run_options()
            if options is None:
                result = encoder.run(list(ENCODER_OUTPUTS), feed)
            else:
                result = encoder.run(list(ENCODER_OUTPUTS), feed, options)
            self._audit(encoder, 'encoder')  # strictly audit the first GPU run
            self._log_timing('encoder_inference', time.perf_counter() - t0)
            return tuple(np.ascontiguousarray(x, dtype=np.float32) for x in result)
        except Exception:
            # Failed native sessions must not be reused on subsequent strokes.
            if getattr(self, 'encoder', None) is encoder:
                self.encoder = None
            raise
        finally:
            if not getattr(self, '_reuse_encoder_session', True):
                del encoder  # optional old memory-first behavior for A/B

    def set_image(self, image_rgb: np.ndarray) -> None:
        image = np.asarray(image_rgb, dtype=np.uint8)
        self._features = self._encode(image)
        self._original_hw = tuple(image.shape[:2])

    def _run_options(self):
        if not self.gpu:
            return None
        from ..app_config import APP_CONFIG
        if not APP_CONFIG.get('sam', {}).get('m71_shrink_gpu_arena_after_run', True):
            return None
        opts = ort.RunOptions()
        opts.add_run_config_entry(
            'memory.enable_memory_arena_shrinkage',
            'gpu:' + self.device.split(':', 1)[1],
        )
        return opts

    def _decode(self, features, original_hw, point_coords, point_labels):
        coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
        if not len(coords) or len(coords) != len(labels) or not np.isin(labels, [-1, 0, 1, 2, 3]).all():
            raise ValueError('HQ3 ONNX: point_coords/labels không hợp lệ.')
        feed = dict(zip(ENCODER_OUTPUTS, features))
        feed['point_coords'] = transform_hq3_points(coords, original_hw)[None]
        feed['point_labels'] = labels[None]
        options = self._run_options()
        t0 = time.perf_counter()
        if options is None:
            result = self.decoder.run(list(DECODER_OUTPUTS), feed)
        else:
            result = self.decoder.run(list(DECODER_OUTPUTS), feed, options)
        self._audit(self.decoder, 'decoder')
        self._log_timing('decoder_inference', time.perf_counter() - t0)
        logits = np.asarray(result[0], dtype=np.float32)
        scores = np.asarray(result[1], dtype=np.float32).reshape(-1)
        if logits.shape != (1, 1, 256, 256) or scores.shape != (1,):
            raise RuntimeError(f'HQ3 ONNX sai output: logits={logits.shape}, scores={scores.shape}')
        h, w = original_hw
        upsample = cv2.resize(logits[0, 0], (w, h), interpolation=cv2.INTER_LINEAR)
        return (upsample > 0.0)[None], scores, logits[0]

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        if self._features is None:
            raise RuntimeError('HQ3 ONNX: cần set_image trước predict.')
        coords = (np.empty((0, 2), dtype=np.float32) if point_coords is None
                  else np.asarray(point_coords, dtype=np.float32).reshape(-1, 2))
        labels = (np.empty((0,), dtype=np.int32) if point_labels is None
                  else np.asarray(point_labels, dtype=np.int32).reshape(-1))
        if len(coords) != len(labels):
            raise ValueError('HQ3 ONNX: prompt coordinates/labels không khớp.')
        if box is not None:
            xyxy = np.asarray(box, dtype=np.float32).reshape(4)
            if not (xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]):
                raise ValueError('HQ3 ONNX: box không hợp lệ.')
            # SAM2's learned corner embeddings are labels 2 (top-left) and 3
            # (bottom-right). The verified decoder graph takes dynamic N points;
            # no new checkpoints/ONNX export are required for these tokens.
            coords = np.concatenate([coords, xyxy.reshape(2, 2)], axis=0)
            labels = np.concatenate([labels, np.array([2, 3], np.int32)])
        _ = multimask_output  # HQ3 trained for one HQ mask
        return self._decode(self._features, self._original_hw, coords, labels)

    def predict_crop(self, image_rgb, crop_xyxy, point_coords, point_labels):
        image = np.asarray(image_rgb, dtype=np.uint8)
        h, w = image.shape[:2]
        x0, y0, x1, y1 = map(int, crop_xyxy)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('HQ3 ONNX crop ngoài ảnh.')
        coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        local = coords - np.array([x0, y0], np.float32)
        if not ((local[:, 0] >= 0).all() and (local[:, 0] < x1-x0).all()
                and (local[:, 1] >= 0).all() and (local[:, 1] < y1-y0).all()):
            raise ValueError('HQ3 ONNX: điểm click ngoài crop.')
        # Independent crop embedding, without overwriting current full-frame features.
        feats = self._encode(image[y0:y1, x0:x1])
        mask, scores, _ = self._decode(feats, (y1-y0, x1-x0), local, point_labels)
        full = np.zeros((1, h, w), dtype=bool)
        full[0, y0:y1, x0:x1] = mask[0]
        return full, scores

    def close(self):
        # Called after the Guided QThread has finished, at Edit exit.
        # Nothing is kept between Edit sessions unless the existing app
        # config explicitly sets hq2_keep_model_loaded=true.
        self._features = None
        self._original_hw = None
        self.encoder = None
        self.decoder = None
