"""Standalone ViTMatte-SAM ONNX matting for video, run after HQ-SAM2 has exited.

Protocol on stdout is ASCII-safe JSON; input and reports use UTF-8. No silent
CPU fallback. Input ONNX contract: float32 [1,4,768,768], alpha [1,1,768,768].
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
try:
    from .video_diagnostics import REPORT_VERSION, dump_json, dump_csv, ram_mib
except ImportError:
    from video_diagnostics import REPORT_VERSION, dump_json, dump_csv, ram_mib

PREFIX = '__ANIBGRMV_VIDEO_MATTING__'


def emit(event):
    print(PREFIX + json.dumps(event, ensure_ascii=True), flush=True)


def model_digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def trimap_input(rgb, coarse):
    """Training-notebook preprocessing: RGB pad=114; erode=17, dilate=33; 768 letterbox."""
    h, w = coarse.shape
    if rgb.shape[:2] != coarse.shape or rgb.shape[2] != 3:
        raise ValueError('ViTMatte-SAM RGB và mask khác kích thước.')
    nw = max(1, round(w * min(768/w, 768/h)))
    nh = max(1, round(h * min(768/w, 768/h)))
    x, y = (768-nw)//2, (768-nh)//2
    patch = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    padded = np.full((768,768,3), 114, np.uint8)
    padded[y:y+nh, x:x+nw] = patch
    small = cv2.resize(np.asarray(coarse > 127, np.uint8), (nw,nh), interpolation=cv2.INTER_NEAREST)
    mm = np.zeros((768,768),np.uint8)
    mm[y:y+nh,x:x+nw] = small
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33))
    sure_fg = cv2.erode(mm, erode)>0
    sure_bg = cv2.dilate(mm, dilate)==0
    trimap = np.full((768,768),128,np.uint8)
    trimap[sure_fg] = 255
    trimap[sure_bg] = 0
    img = padded.astype(np.float32)/255.0
    img = (img-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
    tensor = np.concatenate([np.moveaxis(img,-1,0), trimap[None].astype(np.float32)/255],axis=0)[None]
    return np.ascontiguousarray(tensor,np.float32), (x,y,nw,nh)


def get_alpha(rgb, coarse, session, input_name):
    if not np.any(coarse > 127):
        return np.zeros(coarse.shape, np.uint16)
    tensor, (x,y,nw,nh) = trimap_input(rgb,coarse)
    alpha = np.asarray(session.run(None,{input_name:tensor})[0])
    if alpha.shape != (1,1,768,768):
        raise RuntimeError(f'ViTMatte-SAM ONNX output sai shape: {alpha.shape}')
    a = alpha[0,0]
    if not np.isfinite(a).all():
        raise RuntimeError('ViTMatte-SAM trả alpha NaN/Inf')
    resized = cv2.resize(np.clip(a[y:y+nh,x:x+nw],0,1),
                         (coarse.shape[1],coarse.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.rint(resized*65535),0,65535).astype(np.uint16)


def run(job_path):
    import onnxruntime as ort
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    model = Path(job['model_onnx'])
    report_dir = Path(job['report_dir']); report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(job['cache_dir']); cache_dir.mkdir(parents=True, exist_ok=True)
    frame_count = int(job['frame_count'])
    frame_paths = [Path(p) for p in job['mask_paths']]
    source_indices = [int(i) for i in job['source_frame_indices']]
    report_path = report_dir/'matting_report.json'
    report = {'version':REPORT_VERSION,'status':'RUNNING','frames':frame_count,
              'compute_provider':'CUDAExecutionProvider', 'cache_hits':0,
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
        options.enable_profiling = True
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        init_at = time.perf_counter()
        session = ort.InferenceSession(str(model), sess_options=options,
                                       providers=['CUDAExecutionProvider'])
        if session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('ViTMatte-SAM không được thực thi bằng CUDAExecutionProvider.')
        session.disable_fallback()
        inputs = session.get_inputs(); outputs = session.get_outputs()
        if len(inputs)!=1 or len(outputs)!=1 or inputs[0].type!='tensor(float)' or inputs[0].shape != [1,4,768,768]:
            raise RuntimeError('Sai ONNX input ViTMatte-SAM, yêu cầu float32 [1,4,768,768].')
        input_name=inputs[0].name
        report['session_load_sec']=round(time.perf_counter()-init_at,3)
        emit({'event':'progress','percent':1,'message':'ViTMatte-SAM CUDA sẵn sàng'})
        cap=cv2.VideoCapture(job['source_video'])
        if not cap.isOpened():
            raise RuntimeError('Không đọc được video nguồn để chạy ViTMatte-SAM.')
        source_index=-1
        timings=[]
        alpha_list=[]
        first_cache_key={}
        for fi,(src_i,mask_path) in enumerate(zip(source_indices,frame_paths)):
            at=time.perf_counter()
            frame=None
            while source_index<src_i:
                ok, frame=cap.read()
                if not ok:
                    raise RuntimeError(f'Không decode được source frame {src_i} cho ViTMatte-SAM.')
                source_index+=1
            coarse=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
            if coarse is None:
                raise FileNotFoundError('Thiếu binary mask '+str(mask_path))
            h,w=frame.shape[:2]
            if coarse.shape!=(h,w):
                coarse=cv2.resize(coarse,(w,h),interpolation=cv2.INTER_NEAREST)
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            key=hashlib.sha256(model_hash.encode('ascii')+rgb.tobytes()+coarse.tobytes()).hexdigest()
            output=cache_dir/(key+'.png')
            if output.is_file():
                candidate=cv2.imread(str(output),cv2.IMREAD_UNCHANGED)
                if candidate is not None and candidate.dtype==np.uint16 and candidate.shape==coarse.shape:
                    report['cache_hits']+=1
                    if key in first_cache_key:
                        report['duplicate_reuses']+=1
                else:
                    output.unlink(missing_ok=True)
            if not output.is_file():
                alpha=get_alpha(rgb,coarse,session,input_name)
                temp=cache_dir/(key+'.partial.png')
                if not cv2.imwrite(str(temp),alpha,[cv2.IMWRITE_PNG_COMPRESSION,2]):
                    raise OSError('Không ghi được ViTMatte-SAM alpha '+str(temp))
                temp.replace(output)
                report['computed']+=1
            first_cache_key[key]=fi
            alpha_list.append(str(output))
            timings.append({'frame':fi,'seconds':round(time.perf_counter()-at,4),
                            'cache_hit':int(key in first_cache_key and first_cache_key[key]!=fi),
                            'process_rss_mib':ram_mib() if fi%8==0 else None})
            emit({'event':'progress','percent':min(99,round((fi+1)*100/frame_count)),
                  'done':fi+1,'total':frame_count,'message':f'ViTMatte-SAM alpha {fi+1}/{frame_count}'})
            if fi%8==0:
                report['memory_samples'].append({'frame':fi,'elapsed_sec':round(time.perf_counter()-started,3),
                                                 'rss_mib':ram_mib()})
        cap.release()
        profile = Path(session.end_profiling())
        if profile.is_file():
            final=report_dir/'matting_onnx_profile.json'
            profile.replace(final)
            try:
                entries=json.loads(final.read_text(encoding='utf-8'))
                providers={}; ops={}
                for e in entries:
                    if e.get('cat')!='Node':continue
                    args=e.get('args') or {}
                    provider=args.get('provider','unknown')
                    providers[provider]=providers.get(provider,0)+1
                    op=args.get('op_name','unknown')
                    ops[op]=ops.get(op,0)+float(e.get('dur',0))/1000
                report['ort_profile']={'provider_node_count':providers,
                                       'op_total_milliseconds':{k:round(v,2) for k,v in ops.items()}}
                if report['computed'] and providers.get('CUDAExecutionProvider',0)==0:
                    raise RuntimeError('ViTMatte-SAM không có CUDA node trong ORT profile.')
            except (json.JSONDecodeError,OSError):
                report['ort_profile_error']='Không đọc được ORT profile'
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
