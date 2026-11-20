"""Demo nhận diện khuôn mặt qua webcam (bước 8).

Chạy từ thư mục dự án (DoAn_CV_NangCao), bằng Python của môi trường dự án:

    python ban_giao_nhom/demo/webcam_demo.py                       # webcam 0, hệ thống B3
    python ban_giao_nhom/demo/webcam_demo.py --model B0 --threshold 0.60
    python ban_giao_nhom/demo/webcam_demo.py --source thu_muc_anh --no-window --out ket_qua

Phím khi cửa sổ đang mở:
    q / Esc  thoát          1 / 2 / 3 / 4  đổi hệ thống B0 / B1 / B2 / B3
    + / -    chỉnh ngưỡng   r          nạp lại gallery từ thư mục (reset)

Mỗi khung hình được nhận diện độc lập, không làm mượt theo thời gian: mất mặt thì nhãn mất
ngay. Cosine là độ giống, không phải xác suất đúng. Demo không có chống giả mạo (liveness),
không dùng để mở cửa hay xác thực thật.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'cai_tien_B0_B3'))
from nhan_dien import Viewer, open_rgb, IMAGE_SUFFIXES  # noqa: E402
from cai_tien import LadderRecognizer  # noqa: E402

MODELS = LadderRecognizer.SYSTEMS
COLORS = {'known': (46, 160, 67), 'unknown': (235, 104, 52), 'problem': (208, 59, 59), 'ink': (255, 255, 255)}
STATUS_TEXT = {'no_face': 'Không tìm thấy mặt', 'multiple_faces': 'Nhiều mặt trong khung hình', 'empty_gallery': 'Gallery trống'}


def webcam_thresholds():
    """Ngưỡng chọn trên ảnh webcam phiên 2 (nếu notebook webcam đã chạy), nếu chưa thì dùng ngưỡng dev LFW."""
    webcam = HERE / 'nguong_webcam.json'
    if webcam.exists():
        data = json.loads(webcam.read_text(encoding='utf-8'))
        values = {m: float(data['thresholds'][m]) for m in MODELS if m in data['thresholds']}
        if set(values) == set(MODELS):
            return values, 'webcam phiên 2'
    return {}, 'dev LFW'


def read_people(folder):
    """<folder>/<mã người>/*.jpg -> {mã người: [ảnh PIL]}."""
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {p.name: [open_rgb(f) for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_SUFFIXES]
            for p in sorted(folder.iterdir()) if p.is_dir()}


def font(size):
    for name in ['arial.ttf', 'C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_source(source, mirror):
    """Sinh (mã khung, ảnh PIL RGB) từ webcam, file video hoặc thư mục ảnh."""
    path = Path(source)
    if path.is_dir():
        for p in sorted(x for x in path.rglob('*') if x.suffix.lower() in IMAGE_SUFFIXES):
            yield p.relative_to(path).as_posix(), open_rgb(p)
        return
    cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f'Không mở được nguồn hình: {source}')
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            yield f'frame_{i:06d}', Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
    finally:
        cap.release()


def annotate(image, result, info, reference):
    """Vẽ khung mặt, nhãn, ảnh tham chiếu và dòng trạng thái lên một bản sao của khung hình."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    big, small = font(max(18, h // 20)), font(max(13, h // 34))
    if result['detector_status'] == 'single_face':
        color = COLORS['known'] if result['status'] == 'recognized' else COLORS['unknown']
        x1, y1, x2, y2 = result['box']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{result['predicted_identity']}  cos {result['top_score']:.2f}"
        tw = draw.textlength(label, font=big)
        ty = max(0, y1 - big.size - 8)
        draw.rectangle([x1, ty, x1 + tw + 12, ty + big.size + 8], fill=color)
        draw.text((x1 + 6, ty + 3), label, fill=COLORS['ink'], font=big)
    else:
        for box in result['boxes']:
            draw.rectangle(box, outline=COLORS['problem'], width=2)
        label = STATUS_TEXT.get(result['detector_status'], result['detector_status'])
        tw = draw.textlength(label, font=big)
        draw.rectangle([10, 10, 22 + tw, 18 + big.size], fill=COLORS['problem'])
        draw.text((16, 13), label, fill=COLORS['ink'], font=big)
    if reference is not None:
        size = max(72, h // 5)
        thumb = Image.fromarray(reference).resize((size, size))
        canvas.paste(thumb, (w - size - 10, 10))
        draw.rectangle([w - size - 10, 10, w - 10, 10 + size], outline=COLORS['known'], width=2)
        draw.text((w - size - 10, 14 + size), 'Ảnh đăng ký', fill=COLORS['ink'], font=small,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    draw.rectangle([0, h - small.size - 12, w, h], fill=(20, 20, 20))
    draw.text((8, h - small.size - 8), info, fill=COLORS['ink'], font=small)
    return canvas


def main():
    ap = argparse.ArgumentParser(description='Demo nhận diện khuôn mặt qua webcam')
    ap.add_argument('--gallery', default=str(ROOT / 'data/raw/webcam/s1_dang_ky'), help='thư mục <mã người>/*.jpg đã đăng ký')
    ap.add_argument('--model', default='B3', choices=MODELS, help='hệ thống nhận diện (mặc định B3)')
    ap.add_argument('--threshold', type=float, help='ngưỡng cosine; mặc định lấy theo model')
    ap.add_argument('--source', default='0', help='số thứ tự webcam, file video hoặc thư mục ảnh')
    ap.add_argument('--device', help="'cuda:0' hoặc 'cpu'; mặc định tự chọn")
    ap.add_argument('--min-face', type=int, help='cạnh mặt nhỏ nhất (pixel) cho MTCNN; mặc định 20')
    ap.add_argument('--no-mirror', action='store_true', help='không lật gương hình webcam')
    ap.add_argument('--no-window', action='store_true', help='không mở cửa sổ (chạy lô trên ảnh/video)')
    ap.add_argument('--out', help='thư mục ghi predictions.csv (và khung hình nếu --save-frames)')
    ap.add_argument('--save-frames', action='store_true')
    ap.add_argument('--max-frames', type=int)
    args = ap.parse_args()

    thresholds, threshold_source = webcam_thresholds()
    rec = LadderRecognizer(args.model, device=args.device, root=ROOT, thresholds=thresholds, min_face_size=args.min_face)
    if args.threshold is not None:
        rec.thresholds[args.model] = args.threshold
        threshold_source = 'đặt tay'

    def enroll():
        report = rec.enroll(read_people(args.gallery))
        print(f"Gallery: {len(rec.names)} người / {len(report)} ảnh từ {args.gallery}")
        for r in report:
            if r['status'] == 'fallback':
                print(f"  bỏ qua {r['person']} ảnh {r['index'] + 1}: không thấy mặt")
        if not rec.names:
            print('  Gallery trống: mọi khuôn mặt sẽ là Unknown. Thu ảnh đăng ký bằng thu_anh_webcam.py trước.')
    enroll()

    out_dir = Path(args.out) if args.out else None
    writer = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        handle = (out_dir / 'predictions.csv').open('w', newline='', encoding='utf-8-sig')
        writer = csv.DictWriter(handle, fieldnames=['query_id', 'status', 'predicted_identity', 'nearest_identity', 'top_score',
                                                    'threshold', 'model', 'detector_status', 'n_faces', 'latency_ms'])
        writer.writeheader()

    is_camera = str(args.source).isdigit()
    viewer = None if args.no_window else Viewer('Nhận diện khuôn mặt (q: thoát)', hidden=os.environ.get('CV_DEMO_HIDE_WINDOW') == '1')
    loop_times, n = [], 0
    last = time.perf_counter()
    for frame_id, image in frame_source(args.source, mirror=is_camera and not args.no_mirror):
        threshold = rec.threshold
        if rec.device.startswith('cuda'):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        r = rec.identify(image)
        matched = r['status'] in ('recognized', 'unknown')
        result = {'query_id': frame_id, 'detector_status': 'single_face' if matched else r['status'], 'n_faces': r['n_faces'],
                  'boxes': r['boxes'], 'box': r['box'], 'status': r['status'], 'predicted_identity': r['name'],
                  'nearest_identity': r['nearest'], 'top_score': r['score'] if matched else '', 'threshold': threshold, 'model': rec.system}
        reference = rec.reference(r['name']) if r['status'] == 'recognized' else None
        if rec.device.startswith('cuda'):
            torch.cuda.synchronize()
        result['latency_ms'] = 1000 * (time.perf_counter() - begin)
        now = time.perf_counter(); loop_times = (loop_times + [now - last])[-30:]; last = now
        if writer:
            writer.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in result.items() if k in writer.fieldnames})
        if viewer or (out_dir and args.save_frames):
            info = (f"{rec.system} · ngưỡng {threshold:.3f} ({threshold_source}) · {len(rec.names)} người · "
                    f"xử lý {result['latency_ms']:.0f} ms · {1 / max(np.mean(loop_times), 1e-6):.0f} FPS")
            shown = annotate(image, result, info, reference)
            if out_dir and args.save_frames:
                target = out_dir / 'frames' / (Path(frame_id).with_suffix('.jpg').as_posix().replace('/', '__'))
                target.parent.mkdir(parents=True, exist_ok=True); shown.save(target, quality=90)
            if viewer:
                keys = viewer.show(shown)
                if Path(args.source).is_dir():
                    time.sleep(0.4)                  # ảnh tĩnh: dừng một chút để kịp nhìn
                stop = False
                for key in keys:
                    if key in ('q', 'Q', 'Escape'):
                        stop = True
                    elif key in ('1', '2', '3', '4'):
                        rec.set_system(MODELS[int(key) - 1])
                    elif key in ('plus', 'equal', 'KP_Add'):
                        rec.thresholds[rec.system] = min(1.0, rec.threshold + 0.01); threshold_source = 'đặt tay'
                    elif key in ('minus', 'underscore', 'KP_Subtract'):
                        rec.thresholds[rec.system] = max(-1.0, rec.threshold - 0.01); threshold_source = 'đặt tay'
                    elif key in ('r', 'R'):
                        enroll()
                if stop:
                    break
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    if writer:
        handle.close()
        print(f'Đã ghi {n} dòng vào {out_dir / "predictions.csv"}')
    if viewer:
        viewer.close()


if __name__ == '__main__':
    main()
