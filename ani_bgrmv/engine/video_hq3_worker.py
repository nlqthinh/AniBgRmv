"""Isolated HQ-SAM2/HQ3 video tracker (run by a Python environment with torch).

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
try:
    from .video_diagnostics import dump_json, dump_csv, ram_mib, qseconds, now
except ImportError:
    from video_diagnostics import dump_json, dump_csv, ram_mib, qseconds, now
from pathlib import Path

PROGRESS_PREFIX = '__ANIBGRMV_HQ3_TRACK__'


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


def run(job_file: Path):
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_hq_video_predictor

    job = json.loads(job_file.read_text(encoding='utf-8'))
    base = Path(job['base_checkpoint'])
    best = Path(job['hq3_decoder_checkpoint'])
    frames_dir = Path(job['frames_dir'])
    output = Path(job['output_dir'])
    count = int(job['frame_count'])
    target_hw = tuple(int(v) for v in job['hw'])
    annotations = {int(k): Path(v) for k, v in job['keyframe_masks'].items()}
    if not torch.cuda.is_available():
        raise RuntimeError('Video Predictor yêu cầu CUDA; không tự chuyển sang CPU.')
    for asset in (base, best):
        if not asset.is_file():
            raise FileNotFoundError('Thiếu checkpoint video HQ3: ' + str(asset))
    if not annotations or not any(p.is_file() for p in annotations.values()):
        raise RuntimeError('Không có keyframe mask hợp lệ.')
    if not hasattr(torch, 'autocast'):
        raise RuntimeError('Phiên bản PyTorch không hỗ trợ autocast CUDA.')

    started = now()
    stages = {}
    frame_runtime = []
    memory_samples = []
    report_path = Path(job.get('worker_report', output / 'track_worker_report.json'))
    worker_report = {'version': 'M87', 'status': 'RUNNING', 'frames_expected': count,
                     'keyframe_indices': sorted(annotations), 'stages_sec': stages,
                     'gpu': None, 'memory_samples': memory_samples}
    def snapshot(stage, frame=None):
        snap = {'stage': stage, 'frame': frame, 'elapsed_sec': qseconds(now()-started),
                'process_rss_mib': ram_mib()}
        try:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            snap.update(cuda_allocated_mib=round(torch.cuda.memory_allocated()/1048576, 1),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved()/1048576, 1),
                        cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1048576, 1),
                        cuda_driver_used_mib=round((total-free)/1048576, 1))
        except Exception as ex:
            snap['gpu_sample_error'] = str(ex)
        memory_samples.append(snap)
        emit({'event': 'metric', 'sample': snap})
        return snap
    worker_report['gpu'] = torch.cuda.get_device_name(0)
    snapshot('before_model_load')
    model_t0 = now()
    emit({'event': 'status', 'stage': 'loading', 'percent': 1,
          'message': 'Đang nạp HQ-SAM2 Video Predictor + HQ3 BEST...'})
    # Follow the notebook exactly: base weights, then replace HQ decoder.
    with torch.inference_mode(False):
        model = build_sam2_hq_video_predictor(
            'configs/sam2.1/sam2.1_hq_hiera_l.yaml', str(base),
            device='cuda', mode='eval',
            hydra_overrides_extra=['++model.add_all_frames_to_correct_as_cond=true'],
        )
        checkpoint = torch.load(str(best), map_location='cpu', weights_only=True)
        if (not isinstance(checkpoint, dict)
                or checkpoint.get('format') != 'HQ2_HQSAM2_decoder_state_v2'
                or not isinstance(checkpoint.get('decoder'), dict)):
            raise ValueError('HQ3 BEST không đúng định dạng decoder HQ-SAM2 của app.')
        model.sam_mask_decoder.load_state_dict(checkpoint['decoder'], strict=True)
        del checkpoint
    model.eval()
    stages['load_model'] = qseconds(now()-model_t0)
    snapshot('after_model_load')
    if not callable(getattr(model, 'add_new_mask', None)):
        raise RuntimeError('HQ-SAM2 video predictor hiện tại thiếu add_new_mask.')

    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        state_t0 = now()
        state = model.init_state(
            video_path=str(frames_dir), offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False,
        )
        if int(state['num_frames']) != count:
            raise RuntimeError(f'Video state có {state["num_frames"]} frame, cần {count}.')
        model.reset_state(state)
        stages['init_state'] = qseconds(now()-state_t0)
        snapshot('after_init_state')
        prompt_t0 = now()
        for frame_idx, mask_path in sorted(annotations.items()):
            if not 0 <= frame_idx < count:
                raise ValueError(f'Sai chỉ số keyframe {frame_idx}.')
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != target_hw:
                raise ValueError(f'Mask keyframe {frame_idx} sai kích thước {target_hw}.')
            # A binary mask is a proper SAM2 prompt; do not synthesize clicks
            # from manually drawn strokes. This preserves actual user shape.
            model.add_new_mask(state, frame_idx=frame_idx, obj_id=1, mask=mask >= 128)
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
                if 1 not in ids:
                    raise RuntimeError(f'Không tìm thấy object 1 ở frame {frame_idx}.')
                scores = logits[ids.index(1), 0]
                if not torch.isfinite(scores).all().item():
                    raise RuntimeError(f'Mask logits không hữu hạn ở frame {frame_idx}.')
                frame_t0 = now()
                mask = (scores > 0).to('cpu').numpy().astype(np.uint8) * 255
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
            raise ValueError('Usage: video_hq3_worker.py <job.json>')
        run(Path(sys.argv[1]))
    except BaseException as exc:
        try:
            job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
            where = job.get('worker_report')
            if where:
                dump_json(where, {'version': 'M87', 'status': 'ERROR',
                                  'error': f'{type(exc).__name__}: {exc}',
                                  'traceback': traceback.format_exc()})
        except Exception:
            pass
        emit({'event': 'error', 'message': f'{type(exc).__name__}: {exc}'})
        traceback.print_exc()
        sys.exit(1)
