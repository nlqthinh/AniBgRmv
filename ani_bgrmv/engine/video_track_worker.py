"""Isolated HQ-SAM2 video tracker (run by a Python environment with torch).

Uses the SysCV HQ-SAM2 Video Predictor, its temporal memory and bidirectional
propagation. NOT optical flow. Arguments: path to a trusted JSON job. No network,
model downloads or automatic dependency installation.
"""
from __future__ import annotations

import json
import sys
import time
import statistics
import traceback
from contextlib import contextmanager, nullcontext
try:
    from .video_diagnostics import REPORT_VERSION, dump_json, dump_csv, ram_mib, qseconds, now
except ImportError:
    from video_diagnostics import REPORT_VERSION, dump_json, dump_csv, ram_mib, qseconds, now
from pathlib import Path

PROGRESS_PREFIX = '__ANIBGRMV_VIDEO_TRACK__'
# Decoder checkpoint format written by the HQ-SAM training notebook.
DECODER_FORMAT = 'HQ2_HQSAM2_decoder_state_v2'
# Defaults of SAM2 load_video_frames (init_state does not override them).
SAM2_IMAGE_MEAN = (0.485, 0.456, 0.406)
SAM2_IMAGE_STD = (0.229, 0.224, 0.225)
# video.track_precision = auto: FP16 only if this GPU multiplies FP16 at least
# this much faster than FP32. GTX 16xx/10xx have no usable FP16 tensor cores:
# GTX 1650 measured FP16 0.42 vs FP32 1.76 TFLOPS (App_04 profile_track).
FP16_MIN_SPEEDUP = 1.25


class LazyFrames:
    """Video frames for SAM2, decoded one at a time when the predictor needs them.

    SAM2's own loader keeps every frame as a float32 3x1024x1024 tensor:
    12 MiB per frame, about 2.8 GiB for a 10 s clip. The predictor only uses
    len(images) and images[i]; each item is computed like SAM2's own loader.
    """

    def __init__(self, paths, image_size):
        import torch
        from sam2.utils.misc import _load_img_as_tensor
        self._paths = [str(p) for p in paths]
        self._size = int(image_size)
        self._load = _load_img_as_tensor
        self._mean = torch.tensor(SAM2_IMAGE_MEAN, dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(SAM2_IMAGE_STD, dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        image, _, _ = self._load(self._paths[index], self._size)
        image = image.float()  # SAM2 copies each frame into a float32 tensor, then normalizes
        image -= self._mean
        image /= self._std
        return image


@contextmanager
def lazy_frame_loading(model, frame_paths):
    """While active, model.init_state() receives LazyFrames instead of loading every frame."""
    from PIL import Image
    predictor = sys.modules[type(model).__module__]
    eager = getattr(predictor, 'load_video_frames', None)
    if eager is None:
        raise RuntimeError('Bản SAM2 đang cài không có load_video_frames, không nạp frame lười được.')

    def load_video_frames(*, image_size, **_):  # init_state passes keyword arguments only
        with Image.open(frame_paths[0]) as first:
            width, height = first.size
        return LazyFrames(frame_paths, image_size), height, width

    predictor.load_video_frames = load_video_frames
    try:
        yield
    finally:
        predictor.load_video_frames = eager


def emit(event: dict):
    """Write protocol events in ASCII, independent of Windows console code page.

    JSON escapes non-ASCII characters, and the parent decodes them with
    json.loads(). This protects the progress/error channel even when a user's
    terminal defaults to cp1252, cp1258, or another legacy encoding.
    """
    print(PROGRESS_PREFIX + json.dumps(event, ensure_ascii=True), flush=True)


def _configure_stdio():
    """Allow tracebacks with non-ASCII paths/text to survive redirected pipes."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            reconfigure(encoding='utf-8', errors='backslashreplace')


def load_predictor(base: Path, decoder: Path, device: str):
    """HQ-SAM2 video predictor as in the notebook: base weights, then the fine-tuned decoder."""
    import torch
    from sam2.build_sam import build_sam2_hq_video_predictor
    with torch.inference_mode(False):
        model = build_sam2_hq_video_predictor(
            'configs/sam2.1/sam2.1_hq_hiera_l.yaml', str(base),
            device=device, mode='eval',
            hydra_overrides_extra=['++model.add_all_frames_to_correct_as_cond=true'],
        )
        checkpoint = torch.load(str(decoder), map_location='cpu', weights_only=True)
        if (not isinstance(checkpoint, dict)
                or checkpoint.get('format') != DECODER_FORMAT
                or not isinstance(checkpoint.get('decoder'), dict)):
            raise ValueError('Checkpoint decoder HQ-SAM không đúng định dạng của app.')
        model.sam_mask_decoder.load_state_dict(checkpoint['decoder'], strict=True)
        del checkpoint
    model.eval()
    # Inference only. Weights that require grad are cached by autocast as an
    # extra FP16 copy for the whole Track (~0.45 GB for HQ-SAM2 Large); on a
    # 4 GB GPU that pushes memory into Windows' slow shared GPU memory.
    model.requires_grad_(False)
    if not callable(getattr(model, 'add_new_mask', None)):
        raise RuntimeError('HQ-SAM2 video predictor hiện tại thiếu add_new_mask.')
    return model


def matmul_tflops(device: str, dtype, n: int = 2048, repeats: int = 5) -> float:
    """What a large matrix multiply reaches on this GPU (linear layers are ~80% of Track)."""
    import torch
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        a @ b
    torch.cuda.synchronize(device)
    return 2 * n ** 3 * repeats / (time.perf_counter() - started) / 1e12


def choose_precision(requested: str, device: str) -> tuple[str, dict]:
    """'fp16' or 'fp32' for this Track, and the measurement behind an 'auto' choice."""
    import torch
    if device == 'cpu':
        return 'fp32', {}
    if requested != 'auto':
        return requested, {}
    speed = {'fp16': matmul_tflops(device, torch.float16), 'fp32': matmul_tflops(device, torch.float32)}
    return ('fp16' if speed['fp16'] >= FP16_MIN_SPEEDUP * speed['fp32'] else 'fp32'), speed


def configure_attention(precision: str) -> bool:
    """FP32: let HQ-SAM2 attention use every SDPA kernel, memory-efficient included.

    HQ-SAM2 allows Flash Attention only on Ampere+, so older GPUs get the math
    kernel, which stores the whole attention matrix; it grows with each keyframe
    (5 keyframes: +0.65 GB in the App_05 Track). GTX 1650, FP32 (App_05
    profile_track): same speed, 675 MiB less VRAM, same masks. FP16 keeps
    HQ-SAM2's choice: there the memory-efficient kernel was slower.
    """
    from sam2.modeling.sam import transformer
    transformer.ALLOW_ALL_KERNELS = precision == 'fp32'
    return transformer.ALLOW_ALL_KERNELS


def track_precision(device: str, precision: str):
    """FP16 autocast, or plain FP32 (always FP32 on CPU)."""
    import torch
    if device == 'cpu' or precision == 'fp32':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.float16)


def new_state(model, frame_paths):
    """Video state for the given frames; frames are decoded only when needed (LazyFrames)."""
    with lazy_frame_loading(model, frame_paths):
        return model.init_state(
            video_path=str(frame_paths[0].parent), offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False,
        )


def run(job_file: Path):
    import cv2
    import numpy as np
    import torch

    job = json.loads(job_file.read_text(encoding='utf-8'))
    base = Path(job['base_checkpoint'])
    best = Path(job['decoder_checkpoint'])
    frame_paths = [Path(p) for p in job['frame_paths']]
    output = Path(job['output_dir'])
    count = len(frame_paths)
    target_hw = tuple(int(v) for v in job['hw'])
    annotations = {int(k): Path(v) for k, v in job['keyframe_masks'].items()}
    device = job.get('device', 'cpu')
    requested_precision = job.get('precision', 'auto')
    if requested_precision not in ('auto', 'fp16', 'fp32'):
        raise ValueError(f'video.track_precision không hợp lệ: {requested_precision!r}')
    keyframes_in_memory = int(job.get('keyframes_in_memory', 0))
    if keyframes_in_memory < 0 or keyframes_in_memory == 1:  # SAM2 needs 2+ (or all)
        raise ValueError(f'video.track_keyframes_in_memory không hợp lệ: {keyframes_in_memory}')
    if device != 'cpu':
        import re
        if not isinstance(device, str) or re.fullmatch(r'cuda:(0|[1-9][0-9]*)', device) is None:
            raise ValueError(f'Thiết bị video không hợp lệ: {device!r}')
        if not torch.cuda.is_available():
            raise RuntimeError(f'Đã chọn {device} nhưng PyTorch không có CUDA; không tự dùng CPU.')
        if int(device.split(':')[1]) >= torch.cuda.device_count():
            raise RuntimeError(f'Không tìm thấy GPU {device} trong PyTorch.')
        torch.cuda.set_device(device)
    for asset in (base, best):
        if not asset.is_file():
            raise FileNotFoundError('Thiếu checkpoint HQ-SAM cho video: ' + str(asset))
    if not frame_paths or not all(p.is_file() for p in frame_paths):
        raise FileNotFoundError('Thiếu frame video tạm cho Track.')
    if not annotations or not any(p.is_file() for p in annotations.values()):
        raise RuntimeError('Không có keyframe mask hợp lệ.')

    started = now()
    stages = {}
    frame_runtime = []
    memory_samples = []
    report_path = Path(job.get('worker_report', output / 'track_worker_report.json'))
    worker_report = {'version': REPORT_VERSION, 'status': 'RUNNING', 'frames_expected': count,
                     'keyframe_indices': sorted(annotations), 'stages_sec': stages,
                     'device': device, 'gpu': None, 'memory_samples': memory_samples}
    def snapshot(stage, frame=None):
        snap = {'stage': stage, 'frame': frame, 'elapsed_sec': qseconds(now()-started),
                'process_rss_mib': ram_mib()}
        if device != 'cpu':
            try:
                torch.cuda.synchronize(device)
                free, total = torch.cuda.mem_get_info(device)
                snap.update(cuda_allocated_mib=round(torch.cuda.memory_allocated(device)/1048576, 1),
                            cuda_reserved_mib=round(torch.cuda.memory_reserved(device)/1048576, 1),
                            cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated(device)/1048576, 1),
                            cuda_driver_used_mib=round((total-free)/1048576, 1))
            except Exception as ex:
                snap['gpu_sample_error'] = str(ex)
        memory_samples.append(snap)
        emit({'event': 'metric', 'sample': snap})
        return snap
    worker_report['gpu'] = torch.cuda.get_device_name(device) if device != 'cpu' else None
    snapshot('before_model_load')
    model_t0 = now()
    emit({'event': 'status', 'stage': 'loading', 'percent': 1,
          'message': 'Đang nạp HQ-SAM2 Video Predictor + decoder đã fine-tune...'})
    model = load_predictor(base, best, device)
    # Each frame attends to every keyframe (-1) or to the N closest in time.
    # The cost grows with keyframes x objects (App_13 report, GTX 1650:
    # 15 keyframes x 2 objects ~ 58% of each frame); 0 keeps HQ-SAM2's default.
    model.max_cond_frames_in_attn = keyframes_in_memory or -1
    stages['load_model'] = qseconds(now()-model_t0)
    snapshot('after_model_load')
    precision, matmul_speed = choose_precision(requested_precision, device)
    worker_report.update(precision=precision, precision_requested=requested_precision,
                         matmul_tflops=matmul_speed, keyframes_in_memory=keyframes_in_memory,
                         attention_all_kernels=configure_attention(precision))

    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), track_precision(device, precision):
        state_t0 = now()
        state = new_state(model, frame_paths)
        if int(state['num_frames']) != count:
            raise RuntimeError(f'Video state có {state["num_frames"]} frame, cần {count}.')
        model.reset_state(state)
        stages['init_state'] = qseconds(now()-state_t0)
        snapshot('after_init_state')
        prompt_t0 = now()
        keyframes = {}
        for frame_idx, mask_path in sorted(annotations.items()):
            if not 0 <= frame_idx < count:
                raise ValueError(f'Sai chỉ số keyframe {frame_idx}.')
            labels = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if labels is None or labels.shape != target_hw:
                raise ValueError(f'Mask keyframe {frame_idx} sai kích thước {target_hw}.')
            keyframes[frame_idx] = labels
        # Keyframes are label maps (0 = no object, k = Object k). Each object
        # gets its own SAM2 memory; on every keyframe it is prompted with its
        # pixels there, an empty mask meaning "not in this frame".
        objects = sorted({int(v) for labels in keyframes.values() for v in np.unique(labels) if v})
        if not objects:
            raise RuntimeError('Keyframe chưa có object nào.')
        for frame_idx, labels in keyframes.items():
            for object_id in objects:
                # A mask is a proper SAM2 prompt; do not synthesize clicks
                # from manually drawn strokes. This preserves actual user shape.
                model.add_new_mask(state, frame_idx=frame_idx, obj_id=object_id,
                                   mask=labels == object_id)
            # Build this keyframe's memory now: SAM2 caches the image features
            # of the last frame only, so doing it in propagate_in_video ran the
            # image encoder on every keyframe again (App_13: 15 keyframes, +25 s).
            model.propagate_in_video_preflight(state)
        worker_report['objects'] = objects
        stages['add_keyframes'] = qseconds(now()-prompt_t0)
        snapshot('after_keyframes')
        track_t0 = now()
        anchor = min(annotations)
        directions = ([(anchor, True, anchor+1), (anchor, False, count)] if anchor
                      else [(0, False, count)])
        seen = set()
        for start, reverse, length in directions:
            iterator = iter(model.propagate_in_video(
                state, start_frame_idx=start,
                max_frame_num_to_track=length, reverse=reverse))
            while True:
                next_t0 = now()
                try:
                    frame_idx, object_ids, logits = next(iterator)
                except StopIteration:
                    break
                propagation_sec = now()-next_t0
                frame_idx = int(frame_idx)
                ids = [int(x) for x in object_ids]
                if sorted(ids) != objects:
                    raise RuntimeError(f'Frame {frame_idx} trả object {ids}, cần {objects}.')
                scores = logits[:, 0]
                if not torch.isfinite(scores).all().item():
                    raise RuntimeError(f'Mask logits không hữu hạn ở frame {frame_idx}.')
                frame_t0 = now()
                # One object per pixel: the object with the highest positive score.
                best, index = scores.max(dim=0)
                id_table = torch.tensor(ids, dtype=torch.uint8, device=index.device)
                mask = torch.where(best > 0, id_table[index], 0).to('cpu').numpy().astype(np.uint8)
                transfer_sec = now()-frame_t0
                if mask.shape != target_hw:
                    raise RuntimeError(f'Mask frame {frame_idx}: {mask.shape} != {target_hw}.')
                write_t0 = now()
                if not cv2.imwrite(str(output / f'{frame_idx:05d}.png'), mask):
                    raise OSError(f'Không thể lưu mask frame {frame_idx}.')
                frame_runtime.append({'frame': frame_idx, 'reverse': bool(reverse),
                                      'propagate_sec': qseconds(propagation_sec),
                                      'transfer_to_cpu_sec': qseconds(transfer_sec),
                                      'png_write_sec': qseconds(now()-write_t0),
                                      'foreground_ratio': round(float(np.mean(mask > 0)), 6)})
                seen.add(frame_idx)
                if len(frame_runtime) == 1 or len(frame_runtime) % int(job.get('sample_every', 8)) == 0:
                    snapshot('track_frame', frame_idx)
                emit({'event': 'progress', 'percent': min(99, 5 + (94*len(seen)//count)),
                      'done': len(seen), 'total': count,
                      'message': f'HQ-SAM2 Track {len(seen)}/{count} frame'})

    missing = sorted(set(range(count)) - seen)
    if missing:
        raise RuntimeError(f'Video Predictor chưa trả đủ frame: {missing[:12]}')
    stages['propagate_and_save'] = qseconds(now()-track_t0)
    snapshot('after_track')
    worker_report.update(status='OK', total_sec=qseconds(now()-started),
                         frame_runtime_count=len(frame_runtime), unique_frames=len(seen),
                         track_frames_perf={
                            'propagate_sec_sum':qseconds(sum(x['propagate_sec'] for x in frame_runtime)),
                            'propagate_sec_median':qseconds(statistics.median(x['propagate_sec'] for x in frame_runtime)),
                            'gpu_to_cpu_sec_sum':qseconds(sum(x['transfer_to_cpu_sec'] for x in frame_runtime)),
                            'png_write_sec_sum':qseconds(sum(x['png_write_sec'] for x in frame_runtime)),
                         })
    dump_csv(report_path.with_name('track_frames.csv'), frame_runtime)
    dump_json(report_path, worker_report)
    emit({'event': 'complete', 'percent': 100, 'frames': count,
          'seconds': round(time.perf_counter() - started, 3),
          'message': 'HQ-SAM2 Video Predictor hoàn tất'})


if __name__ == '__main__':
    _configure_stdio()
    try:
        if len(sys.argv) != 2:
            raise ValueError('Usage: video_track_worker.py <job.json>')
        run(Path(sys.argv[1]))
    except BaseException as exc:
        try:
            job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
            where = job.get('worker_report')
            if where:
                dump_json(where, {'version': REPORT_VERSION, 'status': 'ERROR',
                                  'error': f'{type(exc).__name__}: {exc}',
                                  'traceback': traceback.format_exc()})
        except Exception:
            pass
        emit({'event': 'error', 'message': f'{type(exc).__name__}: {exc}'})
        traceback.print_exc()
        sys.exit(1)
