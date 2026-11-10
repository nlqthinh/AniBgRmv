"""Standalone ViTMatte-SAM ONNX matting for video, run after HQ-SAM2 has exited.

Protocol on stdout is ASCII-safe JSON; input and reports use UTF-8. No silent
CPU fallback. Input ONNX contract: float32 [1,4,768,768], alpha [1,1,768,768].
Preprocessing is matting_ops (the same code as Guided Edit), so image and
video never drift apart from the M71 notebook.

Alpha files are 8-bit PNG, as the image editor's alpha and what the 8-bit
MP4/PNG output can show. While the GPU mattes a frame, a thread decodes the
next ones and another writes the PNGs (App_15: the 16-bit PNG write alone took
~130 ms of every frame).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
try:
    from . import matting_ops as ops
    from .video_diagnostics import REPORT_VERSION, dump_json, dump_csv, ram_mib
except ImportError:  # run as a plain script by the external Python
    import matting_ops as ops
    from video_diagnostics import REPORT_VERSION, dump_json, dump_csv, ram_mib

PREFIX = '__ANIBGRMV_VIDEO_MATTING__'
ALPHA_PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
READ_AHEAD = 2  # frames decoded while the GPU works


def emit(event):
    print(PREFIX + json.dumps(event, ensure_ascii=True), flush=True)


def model_digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def matting_input(rgb, coarse):
    """One frame + its binary mask -> (pixel_values, letterbox meta), as in the M71 notebook."""
    if rgb.shape[:2] != coarse.shape or rgb.shape[2] != 3:
        raise ValueError('ViTMatte-SAM RGB và mask khác kích thước.')
    letterbox, _, meta = ops.letterbox_rgb(Image.fromarray(rgb), ops.LETTERBOX_SIZE,
                                           ops.SAM_MATTING_INTERPOLATION)
    trimap = ops.mask_trimap(ops.to_letterbox(coarse > 127, meta, cv2.INTER_NEAREST))
    return ops.vitmatte_input(letterbox, trimap), meta


def get_alpha(rgb, coarse, session, input_name):
    """8-bit alpha at the frame's size, as Guided Edit's (ops.restore_from_letterbox)."""
    if not np.any(coarse > 127):
        return np.zeros(coarse.shape, np.uint8)
    tensor, meta = matting_input(rgb, coarse)
    alpha = np.asarray(session.run(None,{input_name:tensor})[0])
    if alpha.shape != ops.ALPHA_OUTPUT_SHAPE:
        raise RuntimeError(f'ViTMatte-SAM ONNX output sai shape: {alpha.shape}')
    if not np.isfinite(alpha).all():
        raise RuntimeError('ViTMatte-SAM trả alpha NaN/Inf')
    return ops.restore_from_letterbox(alpha[0, 0], meta)


def job_frames(cap, source_indices, mask_paths, model_hash):
    """(rgb, coarse, cache key) of each job frame, decoding the video once, in
    order; frames in between are only grabbed (decoded, not converted)."""
    next_index = 0
    for src_i, mask_path in zip(source_indices, mask_paths):
        ok = all(cap.grab() for _ in range(src_i - next_index))
        ok, frame = cap.read() if ok else (False, None)
        if not ok:
            raise RuntimeError(f'Không decode được source frame {src_i} cho ViTMatte-SAM.')
        next_index = src_i + 1
        coarse = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if coarse is None:
            raise FileNotFoundError('Thiếu binary mask ' + str(mask_path))
        h, w = frame.shape[:2]
        if coarse.shape != (h, w):
            coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        key = hashlib.sha256(model_hash.encode('ascii'))
        key.update(rgb)
        key.update(coarse)
        yield rgb, coarse, key.hexdigest()


def read_ahead(pool, items, depth=READ_AHEAD):
    """``items`` produced by ``pool`` (one thread, so in order) ``depth`` items ahead."""
    items = iter(items)
    pending = deque(pool.submit(next, items, None) for _ in range(depth))
    while (item := pending.popleft().result()) is not None:
        pending.append(pool.submit(next, items, None))
        yield item


def cached_alpha(path, shape) -> bool:
    """A usable alpha from an earlier Export of this clip (a damaged one is removed)."""
    if not path.is_file():
        return False
    alpha = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if alpha is not None and alpha.dtype == np.uint8 and alpha.shape == shape:
        return True
    path.unlink(missing_ok=True)
    return False


def save_alpha(alpha, path):
    temp = path.with_name(path.stem + '.partial.png')
    if not cv2.imwrite(str(temp), alpha, ALPHA_PNG):
        raise OSError('Không ghi được ViTMatte-SAM alpha ' + str(temp))
    temp.replace(path)


def first_run_profile(session, report_dir):
    """End ORT profiling after the first frame and prove it ran on CUDA.

    Profiling every frame would slow the export and write a huge JSON.
    """
    final = Path(report_dir) / 'matting_onnx_profile.json'
    Path(session.end_profiling()).replace(final)
    providers, ops_ms = {}, {}
    for event in json.loads(final.read_text(encoding='utf-8')):
        if event.get('cat') != 'Node':
            continue
        args = event.get('args') or {}
        provider = args.get('provider', 'unknown')
        providers[provider] = providers.get(provider, 0) + 1
        op = args.get('op_name', 'unknown')
        ops_ms[op] = ops_ms.get(op, 0) + float(event.get('dur', 0)) / 1000
    if providers.get('CUDAExecutionProvider', 0) == 0:
        raise RuntimeError('ViTMatte-SAM không có CUDA node trong ORT profile.')
    return {'provider_node_count': providers,
            'op_total_milliseconds': {k: round(v, 2) for k, v in ops_ms.items()}}


def run(job_path):
    import onnxruntime as ort
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    model = Path(job['model_onnx'])
    report_dir = Path(job['report_dir']); report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(job['cache_dir']); cache_dir.mkdir(parents=True, exist_ok=True)
    provider_options = job.get('cuda_provider_options')
    if not isinstance(provider_options, dict) or not str(provider_options.get('device_id', '')).isdecimal():
        raise ValueError('Thiếu CUDA device_id hợp lệ cho ViTMatte-SAM video.')
    frame_count = int(job['frame_count'])
    frame_paths = [Path(p) for p in job['mask_paths']]
    source_indices = [int(i) for i in job['source_frame_indices']]
    report_path = report_dir/'matting_report.json'
    report = {'version':REPORT_VERSION,'status':'RUNNING','frames':frame_count,
              'compute_provider':'CUDAExecutionProvider', 'device_id':provider_options['device_id'], 'cache_hits':0,
              'computed':0,'duplicate_reuses':0,'frames_detail':[],'memory_samples':[]}
    started=time.perf_counter()
    dump_json(report_path,report)
    try:
        if not model.is_file():
            raise FileNotFoundError('Thiếu ViTMatte-SAM ONNX: '+str(model))
        if len(frame_paths)!=frame_count or len(source_indices)!=frame_count:
            raise ValueError('ViTMatte-SAM frame_count khác số mask hoặc frame nguồn.')
        # Import Torch to load the compatible cuDNN used by this Python/GTX1650
        # environment. Never download or install dependencies at runtime.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ViTMatte-SAM yêu cầu onnxruntime-gpu/CUDAExecutionProvider, không chuyển CPU.')
        model_hash = model_digest(model)
        report['model_sha256'] = model_hash
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_profiling = True  # first frame only, see first_run_profile
        options.profile_file_prefix = str(report_dir / 'matting_onnx_profile')
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        init_at = time.perf_counter()
        session = ort.InferenceSession(str(model), sess_options=options,
                                       providers=[('CUDAExecutionProvider', provider_options)])
        if session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('ViTMatte-SAM không được thực thi bằng CUDAExecutionProvider.')
        session.disable_fallback()
        inputs = session.get_inputs(); outputs = session.get_outputs()
        expected = ops.model_input_shape(ops.MATTING_CHANNELS)
        if len(inputs)!=1 or len(outputs)!=1 or inputs[0].type!='tensor(float)' or tuple(inputs[0].shape) != expected:
            raise RuntimeError(f'Sai ONNX input ViTMatte-SAM, yêu cầu float32 {list(expected)}.')
        input_name=inputs[0].name
        report['session_load_sec']=round(time.perf_counter()-init_at,3)
        emit({'event':'progress','percent':1,'message':'ViTMatte-SAM CUDA sẵn sàng'})
        cap=cv2.VideoCapture(job['source_video'])
        if not cap.isOpened():
            raise RuntimeError('Không đọc được video nguồn để chạy ViTMatte-SAM.')
        timings=[]
        alpha_list=[]
        this_run=set()  # keys matted (or being written) in this job
        profiled=False
        writes=deque()
        with ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(1) as writer:
            frames=read_ahead(reader, job_frames(cap, source_indices, frame_paths, model_hash))
            for fi,(rgb,coarse,key) in enumerate(frames):
                at=time.perf_counter()
                output=cache_dir/(key+'.png')
                cache_hit = key in this_run or cached_alpha(output, coarse.shape)
                if cache_hit:
                    report['cache_hits']+=1
                    report['duplicate_reuses']+=key in this_run
                else:
                    alpha=get_alpha(rgb,coarse,session,input_name)
                    if not profiled:
                        report['ort_profile']=first_run_profile(session,report_dir)
                        profiled=True
                    writes.append(writer.submit(save_alpha, alpha, output))
                    report['computed']+=1
                while writes and writes[0].done():
                    writes.popleft().result()  # raises a failed write
                this_run.add(key)
                alpha_list.append(str(output))
                timings.append({'frame':fi,'seconds':round(time.perf_counter()-at,4),
                                'cache_hit':int(cache_hit),
                                'process_rss_mib':ram_mib() if fi%8==0 else None})
                emit({'event':'progress','percent':min(99,round((fi+1)*100/frame_count)),
                      'done':fi+1,'total':frame_count,'message':f'ViTMatte-SAM alpha {fi+1}/{frame_count}'})
                if fi%8==0:
                    report['memory_samples'].append({'frame':fi,'elapsed_sec':round(time.perf_counter()-started,3),
                                                     'rss_mib':ram_mib()})
            for write in writes:
                write.result()
        cap.release()
        if not profiled:  # every frame came from the cache
            Path(session.end_profiling()).unlink(missing_ok=True)
        dump_json(report_dir/'matting_alpha_index.json',alpha_list)
        dump_csv(report_dir/'matting_frame_times.csv',timings)
        report.update(status='OK',total_sec=round(time.perf_counter()-started,3),
                      avg_sec_per_frame=round((time.perf_counter()-started)/frame_count,4),
                      alpha_index=str(report_dir/'matting_alpha_index.json'))
        dump_json(report_path,report)
        emit({'event':'complete','percent':100,'message':'ViTMatte-SAM hoàn tất'})
    except BaseException as exc:
        report.update(status='ERROR',error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc(),total_sec=round(time.perf_counter()-started,3))
        dump_json(report_path,report)
        raise


if __name__=='__main__':
    try:
        run(Path(sys.argv[1]))
    except BaseException as exc:
        emit({'event':'error','message':f'{type(exc).__name__}: {exc}'})
        traceback.print_exc()
        sys.exit(1)
