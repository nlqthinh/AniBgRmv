"""HQ-SAM2 Large on ONNX Runtime (no PyTorch at runtime).

tools/export_hqsam_onnx.py splits the public HQ-SAM2 Large base + the
fine-tuned decoder into two graphs:
  encoder(image[1,3,1024,1024]) -> high_res_feats_0, high_res_feats_1, image_embed
  decoder(features, point_coords, point_labels) -> masks[1,1,256,256], iou_predictions

The encoder is ~200x more expensive than the decoder: one encoder pass per
image (or per magnified crop), then every Paint/Box prompt reuses it.
HqSamFeatures can be saved to disk, so a GPU PC encodes in advance and a slow
CPU PC only runs HqSamDecoder (instance annotation, see ani_bgrmv/annotation).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..hqsam_manifest import check_hqsam_manifest
from .onnx_cuda_audit import HQSAM_DECODER_PART
from .onnx_device import OnnxDevice

SAM_SIZE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
ENCODER_PART = 'HQ-SAM encoder'
ENCODER_OUTPUTS = ('high_res_feats_0', 'high_res_feats_1', 'image_embed')
FEATURE_SHAPES = ((1, 32, 256, 256), (1, 64, 128, 128), (1, 256, 64, 64))
DECODER_INPUTS = (*ENCODER_OUTPUTS, 'point_coords', 'point_labels')
DECODER_OUTPUTS = ('masks', 'iou_predictions')
# SAM2 prompt labels: 1 = positive point, 0 = negative point, -1 = padding,
# 2/3 = learned box-corner tokens (top-left / bottom-right).
PROMPT_LABELS = (-1, 0, 1, 2, 3)
BOX_LABELS = np.array([2, 3], np.int32)


def preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """SAM2 1024x1024 bilinear resize + ImageNet normalization, NCHW."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('HQ-SAM cần ảnh RGB uint8 (H,W,3).')
    resized = np.asarray(Image.fromarray(np.ascontiguousarray(image), 'RGB').resize(
        (SAM_SIZE, SAM_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return np.ascontiguousarray(((resized - _MEAN) / _STD).transpose(2, 0, 1)[None])


def transform_points(coords: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """Image pixel (x, y) -> 1024x1024 model space (x and y scale independently)."""
    h, w = original_hw
    xy = np.asarray(coords, dtype=np.float32).reshape(-1, 2).copy()
    xy[:, 0] = xy[:, 0] / float(w) * SAM_SIZE
    xy[:, 1] = xy[:, 1] / float(h) * SAM_SIZE
    return xy


def mask_from_logits(logits: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """Decoder logits [256,256] -> binary mask in the original image pixels."""
    h, w = original_hw
    return cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR) > 0.0


@dataclass(frozen=True)
class HqSamFeatures:
    """Encoder output for one image (or crop) and that image's pixel size."""
    maps: tuple[np.ndarray, np.ndarray, np.ndarray]  # ENCODER_OUTPUTS order
    original_hw: tuple[int, int]

    def half(self) -> HqSamFeatures:
        """FP16 copy (2 bytes per number instead of 4); the decoder still runs in FP32."""
        return HqSamFeatures(tuple(m.astype(np.float16) for m in self.maps), self.original_hw)

    def save(self, path: Path) -> None:
        """Write FP16 maps + original size to an uncompressed .npz (fast to load)."""
        arrays = dict(zip(ENCODER_OUTPUTS, self.half().maps))
        with Path(path).open('wb') as stream:
            np.savez(stream, original_hw=np.asarray(self.original_hw, np.int32), **arrays)

    @classmethod
    def load(cls, path: Path) -> HqSamFeatures:
        with np.load(Path(path), allow_pickle=False) as data:
            missing = [k for k in (*ENCODER_OUTPUTS, 'original_hw') if k not in data.files]
            if missing:
                raise ValueError(f'File features HQ-SAM thiếu {missing}: {path}')
            maps = tuple(np.ascontiguousarray(data[k]) for k in ENCODER_OUTPUTS)
            original_hw = tuple(int(v) for v in data['original_hw'])
        shapes = tuple(m.shape for m in maps)
        if shapes != FEATURE_SHAPES or len(original_hw) != 2 or min(original_hw) <= 0:
            raise ValueError(f'File features HQ-SAM sai kích thước {shapes}, {original_hw}: {path}')
        if not all(np.isfinite(m).all() for m in maps):
            raise ValueError(f'File features HQ-SAM có giá trị không hợp lệ (NaN/Inf): {path}')
        return cls(maps, original_hw)


class HqSamDecoder:
    """Prompts + encoder features -> mask. Light enough for a slow CPU."""

    def __init__(self, decoder_path: Path, device: OnnxDevice):
        self._device = device
        self._session = device.session(Path(decoder_path), HQSAM_DECODER_PART,
                                       DECODER_INPUTS, DECODER_OUTPUTS)

    def logits(self, features: HqSamFeatures, coords, labels) -> tuple[np.ndarray, float]:
        """Low-resolution logits [256,256] (> 0 = selected) and the predicted IoU."""
        coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if not len(coords) or len(coords) != len(labels) or not np.isin(labels, PROMPT_LABELS).all():
            raise ValueError('HQ-SAM: point_coords/labels không hợp lệ.')
        feed = {name: np.asarray(m, dtype=np.float32) for name, m in zip(ENCODER_OUTPUTS, features.maps)}
        feed['point_coords'] = transform_points(coords, features.original_hw)[None]
        feed['point_labels'] = labels[None]
        logits, scores = self._device.run(self._session, HQSAM_DECODER_PART, DECODER_OUTPUTS, feed)
        logits = np.asarray(logits, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if logits.shape != (1, 1, 256, 256) or scores.shape != (1,):
            raise RuntimeError(f'HQ-SAM sai output: logits={logits.shape}, scores={scores.shape}')
        return logits[0, 0], float(scores[0])

    def mask(self, features: HqSamFeatures, coords, labels) -> tuple[np.ndarray, float]:
        """Binary mask in the original image pixels."""
        logits, score = self.logits(features, coords, labels)
        return mask_from_logits(logits, features.original_hw), score


class HqSamOnnxPredictor:
    """Point/box prompts -> one binary mask in original image pixels.

    ``set_image`` caches full-image features for all following ``predict``
    calls. ``predict_crop`` encodes a magnified crop without touching them.
    """

    def __init__(self, encoder_path: Path, decoder_path: Path, device: str, *,
                 digest=None, cpu_threads: int = 2, shrink_gpu_arena: bool = True,
                 log_timing: bool = False, cuda_options: dict | None = None):
        self.encoder_path, self.decoder_path = Path(encoder_path), Path(decoder_path)
        check_hqsam_manifest(self.encoder_path, self.decoder_path, digest=digest)
        self._device = OnnxDevice(device, cpu_threads=cpu_threads, shrink_gpu_arena=shrink_gpu_arena,
                                  log_timing=log_timing, cuda_options=cuda_options)
        self.device = device
        self.uses_gpu = self._device.uses_gpu
        self._features: HqSamFeatures | None = None
        # The encoder session is created on first use (worker thread), then
        # reused for full images and crops until close().
        self._encoder = None
        self._decoder = HqSamDecoder(self.decoder_path, self._device)

    def encode(self, image_rgb: np.ndarray) -> HqSamFeatures:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if self._encoder is None:
            self._encoder = self._device.session(self.encoder_path, ENCODER_PART,
                                                 ('image',), ENCODER_OUTPUTS)
        try:
            result = self._device.run(self._encoder, ENCODER_PART, ENCODER_OUTPUTS,
                                      {'image': preprocess_image(image)})
        except Exception:
            self._encoder = None  # never reuse a session that failed mid-run
            raise
        maps = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in result)
        return HqSamFeatures(maps, tuple(image.shape[:2]))

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._features = self.encode(image_rgb)

    def predict(self, point_coords=None, box=None) -> tuple[np.ndarray, float]:
        """Positive points and/or one xyxy box on the image given to set_image."""
        if self._features is None:
            raise RuntimeError('HQ-SAM: cần set_image trước predict.')
        coords = (np.empty((0, 2), np.float32) if point_coords is None
                  else np.asarray(point_coords, dtype=np.float32).reshape(-1, 2))
        labels = np.ones(len(coords), np.int32)
        if box is not None:
            xyxy = np.asarray(box, dtype=np.float32).reshape(4)
            if not (xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]):
                raise ValueError('HQ-SAM: box không hợp lệ.')
            coords = np.concatenate([coords, xyxy.reshape(2, 2)])
            labels = np.concatenate([labels, BOX_LABELS])
        return self._decoder.mask(self._features, coords, labels)

    def predict_crop(self, image_rgb, crop_xyxy, point_coords) -> tuple[np.ndarray, float]:
        """Encode only ``crop_xyxy`` (higher effective resolution for tiny objects)."""
        image = np.asarray(image_rgb, dtype=np.uint8)
        h, w = image.shape[:2]
        x0, y0, x1, y1 = map(int, crop_xyxy)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('HQ-SAM crop nằm ngoài ảnh.')
        local = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2) - np.array([x0, y0], np.float32)
        if not ((local[:, 0] >= 0).all() and (local[:, 0] < x1 - x0).all()
                and (local[:, 1] >= 0).all() and (local[:, 1] < y1 - y0).all()):
            raise ValueError('HQ-SAM: điểm prompt nằm ngoài crop.')
        features = self.encode(image[y0:y1, x0:x1])
        crop_mask, score = self._decoder.mask(features, local, np.ones(len(local), np.int32))
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = crop_mask
        return mask, score

    def close(self) -> None:
        self._features = None
        self._encoder = None
        self._decoder = None
