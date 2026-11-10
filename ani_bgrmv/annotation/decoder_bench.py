"""Bước 0 speed test of the HQ-SAM decoder on the annotator's CPU PC.

GPU PC, prepare_bench(): encode sample images (timed), check that FP16
storage gives the same masks as FP32, and write a pack with FP16 features,
reference masks, prompts and a copy of the decoder.
CPU PC, run_bench(): time every prompt with the app's CPU settings and
compare each mask with its reference.
Prompts come from the cutout: the largest object's box, a point inside it,
and, when there is a second object, that point plus a negative point on it.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from ..engine.hqsam_onnx import (HqSamDecoder, HqSamFeatures, HqSamOnnxPredictor,
                                 mask_from_logits)
from ..engine.onnx_device import OnnxDevice
from ..hqsam_manifest import sha256_file
from .alpha_split import AlphaSplit, split_alpha
from .dataset_index import read_alpha, read_rgb, sample_entries, usable_entries
from .output_dir import copy_decoder, fresh_output_dir

BENCH_FORMAT = 'anibgrmv-hqsam-decoder-bench-v1'
BENCH_FILE = 'bench.json'
BENCH_THREADS = 2                         # the annotator's Pentium G4400 has 2 cores
REPEATS = 3
PASS_SECONDS, LIMIT_SECONDS = 5.0, 10.0   # requirement: one SAM action <= 5-10 s
MIN_IOU = 0.98                            # same mask as the reference, up to edge rounding


def _inside(mask: np.ndarray) -> list[float]:
    """The pixel deepest inside the mask (farthest from its edges)."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [float(x), float(y)]


def auto_prompts(split: AlphaSplit) -> list[dict]:
    if split.object_count == 0:
        return []
    first = split.labels == 1
    ys, xs = np.nonzero(first)
    point = _inside(first)
    box = [[float(xs.min()), float(ys.min())], [float(xs.max() + 1), float(ys.max() + 1)]]
    prompts = [{'name': 'box', 'coords': box, 'labels': [2, 3]},
               {'name': 'point', 'coords': [point], 'labels': [1]}]
    if split.object_count >= 2:
        prompts.append({'name': 'point_negative', 'coords': [point, _inside(split.labels == 2)],
                        'labels': [1, 0]})
    return prompts


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def prepare_bench(root: Path, index: dict, pack_dir: Path, *, predictor: HqSamOnnxPredictor,
                  reference_decoder: HqSamDecoder, count: int = 20, keep: int = 10,
                  progress=None) -> dict:
    """GPU PC: time the encoder, check FP16, write the pack for the CPU PC."""
    root, pack_dir = Path(root), Path(pack_dir)
    fresh_output_dir(pack_dir, BENCH_FILE, ('features', 'reference'))
    encode_ms, ious, logit_diffs, samples, nonfinite = [], [], [], [], 0
    entries = sample_entries(index, count)
    for done, entry in enumerate(entries, 1):
        prompts = auto_prompts(split_alpha(read_alpha(root / entry['cutout'])))
        if prompts:
            started = time.perf_counter()
            features = predictor.encode(read_rgb(root / entry['image']))
            encode_ms.append((time.perf_counter() - started) * 1000)
            half = features.half()
            nonfinite += sum(int(np.count_nonzero(~np.isfinite(m))) for m in half.maps)
            name = f'sample_{len(samples):02d}'
            sample = {'image': entry['image'], 'features': f'features/{name}.npz', 'prompts': []}
            for prompt in prompts:
                full, _ = reference_decoder.logits(features, prompt['coords'], prompt['labels'])
                small, _ = reference_decoder.logits(half, prompt['coords'], prompt['labels'])
                reference = mask_from_logits(small, half.original_hw)
                ious.append(mask_iou(mask_from_logits(full, features.original_hw), reference))
                logit_diffs.append(float(np.abs(full - small).max()))
                sample['prompts'].append({**prompt, 'reference': f"reference/{name}_{prompt['name']}.png"})
                if len(samples) < keep:
                    Image.fromarray(reference.astype(np.uint8) * 255).save(
                        pack_dir / sample['prompts'][-1]['reference'])
            if len(samples) < keep:
                half.save(pack_dir / sample['features'])
                samples.append(sample)
        if progress:
            progress(done, len(entries))
    if not samples:
        raise ValueError('Không có ảnh mẫu nào có object để thử decoder.')
    decoder_sha256 = copy_decoder(predictor.encoder_path, predictor.decoder_path, pack_dir)
    spec = {'format': BENCH_FORMAT, 'decoder_filename': predictor.decoder_path.name,
            'decoder_sha256': decoder_sha256, 'cpu_threads': BENCH_THREADS,
            'repeats': REPEATS, 'samples': samples}
    (pack_dir / BENCH_FILE).write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    return {'images': len(encode_ms), 'first_encode_ms': encode_ms[0],
            'encode_ms_median': float(np.median(encode_ms[1:] or encode_ms)),
            'prompts': len(ious), 'fp16_min_iou': min(ious), 'fp16_max_logit_diff': max(logit_diffs),
            'fp16_nonfinite': nonfinite,
            'feature_bytes': (pack_dir / samples[0]['features']).stat().st_size,
            'samples': len(samples), 'dataset_images': len(usable_entries(index))}


def _system_info() -> dict:
    return {'cpu': platform.processor() or platform.machine(), 'logical_cores': os.cpu_count(),
            'os': platform.platform(), 'onnxruntime': ort.__version__, 'numpy': np.__version__,
            'python': platform.python_version()}


def bench_verdict(report: dict) -> str:
    if report['min_iou'] < MIN_IOU:
        return 'KHÔNG KHỚP'
    p90 = report['click_ms_p90'] / 1000
    if p90 <= PASS_SECONDS:
        return 'ĐẠT'
    return 'TẠM ĐẠT' if p90 <= LIMIT_SECONDS else 'KHÔNG ĐẠT'


def run_bench(pack_dir: Path) -> dict:
    """CPU PC: time the decoder on the pack and compare with the reference masks."""
    pack_dir = Path(pack_dir)
    spec = json.loads((pack_dir / BENCH_FILE).read_text(encoding='utf-8'))
    if spec.get('format') != BENCH_FORMAT:
        raise ValueError(f'{BENCH_FILE} không phải gói thử tốc độ của bước 0.')
    decoder_path = pack_dir / spec['decoder_filename']
    if not decoder_path.is_file() or sha256_file(decoder_path) != spec['decoder_sha256']:
        raise ValueError('Decoder trong gói bị thiếu hoặc hỏng: hãy chép lại cả gói.')
    started = time.perf_counter()
    decoder = HqSamDecoder(decoder_path, OnnxDevice('cpu', cpu_threads=spec['cpu_threads']))
    session_ms = (time.perf_counter() - started) * 1000
    load_ms, clicks, ious = [], [], []
    for sample in spec['samples']:
        started = time.perf_counter()
        features = HqSamFeatures.load(pack_dir / sample['features'])
        load_ms.append((time.perf_counter() - started) * 1000)
        for prompt in sample['prompts']:
            with Image.open(pack_dir / prompt['reference']) as image:
                reference = np.asarray(image.convert('L')) > 127
            for _ in range(spec['repeats']):
                started = time.perf_counter()
                mask, _ = decoder.mask(features, prompt['coords'], prompt['labels'])
                clicks.append((time.perf_counter() - started) * 1000)
            ious.append(mask_iou(mask, reference))
    later = clicks[1:] or clicks
    report = {'system': _system_info(), 'cpu_threads': spec['cpu_threads'],
              'session_ms': session_ms, 'feature_load_ms_median': float(np.median(load_ms)),
              'first_click_ms': clicks[0], 'click_ms_median': float(np.median(later)),
              'click_ms_p90': float(np.percentile(later, 90)), 'click_ms_max': max(later),
              'clicks': len(clicks), 'min_iou': min(ious)}
    report['verdict'] = bench_verdict(report)
    return report


def encoder_report_text(r: dict) -> str:
    per_image = r['encode_ms_median'] / 1000
    total = r['dataset_images']
    mb = r['feature_bytes'] / 1e6
    fp16_ok = r['fp16_min_iou'] >= MIN_IOU and r['fp16_nonfinite'] == 0
    return '\n'.join([
        f"Ảnh đã encode trên GPU: {r['images']}",
        f"  Lần đầu (gồm nạp model): {r['first_encode_ms'] / 1000:.1f} giây",
        f'  Các lần sau (trung vị): {per_image:.2f} giây/ảnh',
        f'  Ước tính: 1000 ảnh ~ {per_image * 1000 / 60:.0f} phút; '
        f'toàn bộ {total} ảnh ~ {per_image * total / 3600:.1f} giờ',
        '',
        f'Dung lượng features FP16: {mb:.1f} MB/ảnh -> 1000 ảnh ~ {mb:.1f} GB; '
        f'toàn bộ {total} ảnh ~ {mb * total / 1000:.0f} GB',
        f"So FP16 với FP32 ({r['prompts']} lần decode): IoU thấp nhất {r['fp16_min_iou']:.4f}, "
        f"logit lệch tối đa {r['fp16_max_logit_diff']:.3f}, giá trị lỗi NaN/Inf: {r['fp16_nonfinite']}",
        f"  Kết luận: {'FP16 DÙNG ĐƯỢC' if fp16_ok else 'FP16 KHÔNG DÙNG ĐƯỢC - gửi báo cáo cho Claude'}",
        '',
        f"Gói thử tốc độ cho máy người làm thuê: {r['samples']} ảnh.",
    ]) + '\n'


VERDICT_NOTES = {
    'ĐẠT': 'Máy này chạy SAM đủ nhanh cho app gán nhãn.',
    'TẠM ĐẠT': 'Chậm hơn mục tiêu 5 giây nhưng vẫn dưới 10 giây.',
    'KHÔNG ĐẠT': 'Quá 10 giây: cần tối ưu thêm trước khi làm app.',
    'KHÔNG KHỚP': 'Mask khác bản tham chiếu từ máy chủ: gửi báo cáo cho Claude kiểm tra.',
}


def bench_report_text(r: dict) -> str:
    s = r['system']
    return '\n'.join([
        f"KẾT QUẢ: {r['verdict']} - {VERDICT_NOTES[r['verdict']]}",
        f'Yêu cầu: mỗi thao tác SAM <= {PASS_SECONDS:.0f} giây (tối đa {LIMIT_SECONDS:.0f} giây).',
        '',
        f"Nạp decoder (1 lần khi mở app): {r['session_ms'] / 1000:.1f} giây",
        f"Đọc features 1 ảnh: {r['feature_load_ms_median']:.0f} ms",
        f"Click đầu tiên: {r['first_click_ms'] / 1000:.2f} giây",
        f"Mỗi click - trung vị: {r['click_ms_median'] / 1000:.2f} giây; 90% số click dưới: "
        f"{r['click_ms_p90'] / 1000:.2f} giây; chậm nhất: {r['click_ms_max'] / 1000:.2f} giây "
        f"({r['clicks']} lần, {r['cpu_threads']} luồng)",
        f"Mask so với bản tham chiếu của máy chủ: IoU thấp nhất {r['min_iou']:.4f}",
        '',
        f"CPU: {s['cpu']} ({s['logical_cores']} luồng) | {s['os']}",
        f"onnxruntime {s['onnxruntime']} | numpy {s['numpy']} | python {s['python']}",
    ]) + '\n'
