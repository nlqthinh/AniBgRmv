"""Bước 0 command line.

Owner PC (source):   .venv\\Scripts\\python.exe -m ani_bgrmv.annotation.cli <command> ...
Annotator PC (EXE):  AniBGRmv_CPU.exe --annotation-tool <command> ...
The packaged EXE has no console window, so every result (and any error, as
loi_<command>.txt) is written to files next to the requested output.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

from ..app_config import IMAGE_EXTS
from . import dataset_index
from .alpha_split import DEFAULT_MIN_OBJECT_RATIO

OWNER_CPU_THREADS = 4   # decoder threads on the owner PC (Ryzen 5 3550H)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _read_index(path: str) -> dict:
    return dataset_index.check_index(json.loads(Path(path).read_text(encoding='utf-8')), path)


def _progress(label: str):
    def report(done: int, total: int) -> None:
        if done == total or done % 100 == 0:
            print(f'[{label}] {done}/{total}', flush=True)
    return report


def _encoder_device() -> tuple[str, dict]:
    """The GPU picked in the app (GPU Check), with the app's CUDA options."""
    from ..app_config import APP_CONFIG
    from ..compute_selection import current_selection, verify_gpu
    from ..gpu_memory_options import cuda_provider_options
    selection = current_selection()
    if selection['mode'] != 'gpu':
        raise RuntimeError('Cần chạy encoder trên GPU: mở app, GPU Check, chọn GPU NVIDIA rồi chạy lại.')
    index, _ = verify_gpu(selection)
    return f'cuda:{index}', cuda_provider_options(index, settings=APP_CONFIG['compute'])


def _sam_models():
    """Owner PC: HQ-SAM encoder on the app's GPU + decoder on the CPU, the way
    the annotator's PC decodes (same graph, FP16 features)."""
    from ..engine.hqsam_onnx import HqSamDecoder, HqSamOnnxPredictor
    from ..engine.onnx_device import OnnxDevice
    from ..model_registry import model_path
    device, cuda_options = _encoder_device()
    predictor = HqSamOnnxPredictor(model_path('hqsam_encoder'), model_path('hqsam_decoder'),
                                   device, cuda_options=cuda_options)
    return predictor, HqSamDecoder(predictor.decoder_path, OnnxDevice('cpu', cpu_threads=OWNER_CPU_THREADS))


def cmd_index(args) -> None:
    out, summary = Path(args.out), Path(args.out).with_suffix('.txt')
    for old in (out, summary):
        old.unlink(missing_ok=True)  # never leave an older run's result behind
    index = dataset_index.build_index(Path(args.dataset), IMAGE_EXTS, _progress('index'))
    _write(out, json.dumps(index, ensure_ascii=False, indent=1))
    _write(summary, dataset_index.index_summary(index))
    if not dataset_index.usable_entries(index):
        raise ValueError(f'Không có cặp ảnh + cutout nào dùng được. Xem {summary.name}.')


def cmd_compare(args) -> None:
    out = Path(args.out)
    result = dataset_index.compare_indexes(_read_index(args.owner_index), _read_index(args.worker_index))
    _write(out / 'so_khop.txt', dataset_index.compare_summary(result))
    with (out / 'so_khop.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=('image_sha256', 'owner_image', 'worker_image',
                                                     'cutout_same'))
        writer.writeheader()
        writer.writerows(result['matched'])


def _auto_splitter(min_object_ratio: float):
    from .auto_split import auto_split
    predictor, decoder = _sam_models()

    def split_with_sam(rgb, alpha, split):
        features = predictor.encode(rgb).half()
        return auto_split(alpha, split, lambda coords, labels: decoder.logits(features, coords, labels)[0],
                          min_object_ratio)
    return split_with_sam


def cmd_pilot(args) -> None:
    from .pilot import pilot_summary, run_pilot
    splitter = _auto_splitter(args.min_object_ratio) if args.sam else None
    result = run_pilot(Path(args.dataset), _read_index(args.index), Path(args.out), count=args.count,
                       min_object_ratio=args.min_object_ratio, splitter=splitter)
    _write(Path(args.out) / 'bao_cao_pilot.txt', pilot_summary(result))


def cmd_prepare_bench(args) -> None:
    from .decoder_bench import encoder_report_text, prepare_bench
    predictor, decoder = _sam_models()
    report = prepare_bench(Path(args.dataset), _read_index(args.index), Path(args.pack),
                           predictor=predictor, reference_decoder=decoder, count=args.count,
                           progress=_progress('encoder'))
    out = Path(args.out)
    _write(out / 'bao_cao_encoder.json', json.dumps(report, indent=1))
    _write(out / 'bao_cao_encoder.txt', encoder_report_text(report))


def cmd_package_status(args) -> None:
    from .package import groups_overview, package_groups
    print(groups_overview(package_groups(_read_index(args.owner_index)), Path(args.out)), flush=True)


def cmd_package(args) -> None:
    from .package import build_package, check_owner_files, package_groups, package_report_text, plan_groups
    groups = plan_groups(package_groups(_read_index(args.owner_index)), Path(args.out), args.ids)
    if not groups:
        _write(Path(args.out) / 'bao_cao_goi.txt', 'Mọi ID của dataset máy chủ đều đã có gói.\n')
        return
    check_owner_files(Path(args.dataset), groups)
    print(f"[package] Sẽ tạo gói cho ID: {', '.join(g.folder for g in groups)}", flush=True)
    predictor, decoder = _sam_models()
    reports = [build_package(Path(args.dataset), group, Path(args.out), predictor=predictor, decoder=decoder,
                             min_object_ratio=args.min_object_ratio, progress=_progress(group.package))
               for group in groups]
    _write(Path(args.out) / 'bao_cao_goi.txt', package_report_text(reports))


def cmd_bench(args) -> None:
    from .decoder_bench import bench_report_text, run_bench
    report = run_bench(Path(args.pack))
    out = Path(args.out)
    _write(out / 'bao_cao_toc_do.json', json.dumps(report, ensure_ascii=False, indent=1))
    _write(out / 'bao_cao_toc_do.txt', bench_report_text(report))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='annotation-tool', description='AniBgRmv bước 0')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('index', help='quét dataset -> index .json + tóm tắt .txt')
    p.add_argument('dataset')
    p.add_argument('--out', required=True, help='file .json')
    p.set_defaults(run=cmd_index, result_dir=lambda a: Path(a.out).parent)
    p = sub.add_parser('compare', help='so index máy chủ với máy người làm thuê')
    p.add_argument('owner_index')
    p.add_argument('worker_index')
    p.add_argument('--out', required=True, help='thư mục kết quả')
    p.set_defaults(run=cmd_compare, result_dir=lambda a: Path(a.out))
    p = sub.add_parser('pilot', help='tách object trên ảnh mẫu (vùng rời, hoặc --sam)')
    p.add_argument('dataset')
    p.add_argument('--index', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--min-object-ratio', type=float, default=DEFAULT_MIN_OBJECT_RATIO)
    p.add_argument('--sam', action='store_true', help='tách nhóm chạm nhau bằng HQ-SAM (cần GPU)')
    p.set_defaults(run=cmd_pilot, result_dir=lambda a: Path(a.out))
    p = sub.add_parser('prepare-bench', help='GPU: đo encoder, kiểm FP16, tạo gói thử decoder')
    p.add_argument('dataset')
    p.add_argument('--index', required=True)
    p.add_argument('--pack', required=True, help='thư mục gói (bench)')
    p.add_argument('--out', required=True, help='thư mục báo cáo')
    p.add_argument('--count', type=int, default=20)
    p.set_defaults(run=cmd_prepare_bench, result_dir=lambda a: Path(a.out))
    p = sub.add_parser('package', help='GPU: tạo gói gán nhãn cho từng ID (số ở đầu tên file máy chủ)')
    p.add_argument('dataset', help='dataset máy chủ')
    p.add_argument('--owner-index', required=True)
    p.add_argument('--out', required=True, help='thư mục chứa các gói goi_ID_*')
    p.add_argument('--ids', default='', help='ví dụ 4-10 hoặc 1,3,5-8 (bỏ trống = các ID chưa có gói)')
    p.add_argument('--min-object-ratio', type=float, default=DEFAULT_MIN_OBJECT_RATIO)
    p.set_defaults(run=cmd_package, result_dir=lambda a: Path(a.out))
    p = sub.add_parser('package-status', help='các ID của dataset máy chủ và ID nào đã có gói')
    p.add_argument('--owner-index', required=True)
    p.add_argument('--out', required=True, help='thư mục chứa các gói goi_ID_*')
    p.set_defaults(run=cmd_package_status, result_dir=lambda a: Path(a.out))
    p = sub.add_parser('bench', help='CPU: đo tốc độ decoder trên gói')
    p.add_argument('pack')
    p.add_argument('--out', required=True)
    p.set_defaults(run=cmd_bench, result_dir=lambda a: Path(a.out))
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    error_file = args.result_dir(args) / f'loi_{args.command}.txt'
    error_file.unlink(missing_ok=True)
    try:
        args.run(args)
    except Exception:
        _write(error_file, traceback.format_exc())
        print(f'[LỖI] Xem {error_file}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
