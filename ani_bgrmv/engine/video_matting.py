"""Run ViTMatte-SAM ONNX in a separate Python process after HQ-SAM2 Track exits.

The worker can be canceled without leaking ORT CUDA state into the GUI process.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .video_diagnostics import REPORT_VERSION, dump_json, dump_csv, ram_mib, gpu_snapshot, process_cpu_seconds
from .video_pipeline import VideoCanceled, _check_cancel

PREFIX='__ANIBGRMV_VIDEO_MATTING__'


def _sidecar():
    path=Path(__file__).resolve().with_name('video_matting_worker.py')
    if path.is_file():
        return path
    path=Path(getattr(sys,'_MEIPASS',''))/'ani_bgrmv'/'engine'/'video_matting_worker.py'
    if not path.is_file():
        raise FileNotFoundError('Thiếu video_matting_worker.py trong bản build.')
    return path


def run_vitmatte_sam(clip, masks, report_dir, progress=None, cancel=None):
    from ..app_config import VITMATTE_SAM_ONNX_PATH
    from .video_track import _python_runner
    _check_cancel(cancel)
    if not Path(VITMATTE_SAM_ONNX_PATH).is_file():
        raise FileNotFoundError('Thiếu ViTMatte-SAM ONNX cho video: '+str(VITMATTE_SAM_ONNX_PATH))
    diag=Path(report_dir);diag.mkdir(parents=True,exist_ok=True)
    output_dir=diag.parent.parent
    cache_dir=output_dir/'_matting_cache'
    started=time.perf_counter()
    run_report={'version':REPORT_VERSION,'status':'RUNNING','frames':clip.count,
                'action':'matting_export','samples':[]}
    dump_json(diag/'matting_adapter_report.json',run_report)
    with tempfile.TemporaryDirectory(prefix='anibgrmv_matting_') as t:
        staging=Path(t)
        mask_paths=[]
        for idx, m in enumerate(masks):
            _check_cancel(cancel)
            dest=staging/f'{idx:05d}.png'
            if not cv2.imwrite(str(dest),np.asarray(m,np.uint8)):
                raise OSError('Không thể lưu mask '+str(idx))
            mask_paths.append(str(dest))
        job={'model_onnx':VITMATTE_SAM_ONNX_PATH, 'source_video':clip.source,
             'source_frame_indices':clip.source_frame_indices,
             'mask_paths':mask_paths,'frame_count':clip.count,
             'report_dir':str(diag),'cache_dir':str(cache_dir)}
        job_path=staging/'job.json'
        job_path.write_text(json.dumps(job,ensure_ascii=False),encoding='utf-8')
        env=os.environ.copy()
        env['PYTHONIOENCODING']='utf-8:backslashreplace'
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0
        proc=subprocess.Popen([str(_python_runner()),'-u',str(_sidecar()),str(job_path)],
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',bufsize=1,
                              creationflags=creationflags,env=env)
        events=queue.Queue()
        def read_pipe():
            try:
                for line in proc.stdout:
                    events.put(line.rstrip('\r\n'))
            finally:
                events.put(None)
        thread=threading.Thread(target=read_pipe,daemon=True)
        thread.start()
        error=None; complete=False; tail=[]; last_sample=time.perf_counter()
        try:
            while True:
                _check_cancel(cancel)
                try:
                    line=events.get(timeout=.15)
                except queue.Empty:
                    if proc.poll() is not None and not thread.is_alive():
                        break
                    line=''
                if line is None:
                    break
                if line.startswith(PREFIX):
                    try:
                        obj=json.loads(line[len(PREFIX):])
                    except ValueError:
                        continue
                    if obj.get('event')=='error':
                        error=obj.get('message','Unknown matting error')
                    if obj.get('event')=='complete':
                        complete=True
                    if progress and 'percent' in obj:
                        progress(int(obj['percent']),str(obj.get('message','ViTMatte-SAM')))
                elif line.strip():
                    tail.append(line[:400]);tail=tail[-15:]
                if time.perf_counter()-last_sample>3:
                    run_report['samples'].append({'elapsed_sec':round(time.perf_counter()-started,3),
                        'worker_rss_mib':ram_mib(proc.pid),'parent_rss_mib':ram_mib(),
                        'worker_cpu_total_sec':process_cpu_seconds(proc.pid),
                        **gpu_snapshot()})
                    last_sample=time.perf_counter()
            code=proc.wait(timeout=5)
            _check_cancel(cancel)
            if code!=0 or not complete:
                raise RuntimeError('ViTMatte-SAM Video Export thất bại (exit %d): %s' %
                                   (code,error or '\n'.join(tail[-6:]) or 'Không có log.'))
            alpha_files=json.loads((diag/'matting_alpha_index.json').read_text(encoding='utf-8'))
            if len(alpha_files)!=clip.count or not all(Path(f).is_file() for f in alpha_files):
                raise RuntimeError('ViTMatte-SAM trả thiếu alpha cache.')
            run_report.update(status='OK',total_sec=round(time.perf_counter()-started,3))
            dump_json(diag/'matting_adapter_report.json',run_report)
            dump_csv(diag/'matting_process_samples.csv',run_report['samples'])
            return alpha_files
        except BaseException as exc:
            run_report.update(status='CANCELLED' if isinstance(exc,VideoCanceled) else 'ERROR',
                              error=str(exc),total_sec=round(time.perf_counter()-started,3))
            dump_json(diag/'matting_adapter_report.json',run_report)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait(timeout=3)
            thread.join(timeout=1)
