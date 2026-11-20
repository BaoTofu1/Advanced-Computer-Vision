"""Thu ảnh webcam đúng cấu trúc thư mục của HUONG_DAN_THU_ANH_WEBCAM.md.

Chạy từ thư mục dự án (DoAn_CV_NangCao):

    python ban_giao_nhom/demo/thu_anh_webcam.py --phien s1_dang_ky   --ma-nguoi W01 --dieu-kien ro
    python ban_giao_nhom/demo/thu_anh_webcam.py --phien s2_hieu_chinh --ma-nguoi W01 --dieu-kien khau_trang
    python ban_giao_nhom/demo/thu_anh_webcam.py --phien s3_test       --ma-nguoi U03 --dieu-kien kinh

Phím: Space chụp · q / Esc kết thúc.

Ảnh lưu vào data/raw/webcam/<phiên>/<mã người>/<điều kiện>_<số>.jpg (khung hình gốc, không lật
gương, không vẽ gì lên). Mỗi lần chụp ghi một dòng vào data/raw/webcam/nhat_ky_thu_anh.csv.

- Phiên đăng ký (s1) chỉ lưu khi thấy đúng một mặt, vì ảnh này làm gallery.
- Phiên hiệu chỉnh và test lưu cả khi detector không thấy mặt: đó là lỗi thật của hệ thống
  và phải nằm trong kết quả. Cửa sổ sẽ báo để người chụp biết.
- Người lạ (vai trò unknown) không được có ảnh trong s1_dang_ky.
- Nhớ điền data/raw/webcam/nguoi_tham_gia.csv và xin đồng ý trước khi chụp.
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import sys

import cv2
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from nhan_dien import FacePipeline, Viewer  # noqa: E402
from webcam_demo import font  # noqa: E402

SESSIONS = ['s1_dang_ky', 's2_hieu_chinh', 's3_test']
CONDITIONS = ['ro', 'khau_trang', 'kinh', 'tay_che', 'nghieng', 'toi']


def main():
    ap = argparse.ArgumentParser(description='Thu ảnh webcam cho đồ án')
    ap.add_argument('--phien', required=True, choices=SESSIONS)
    ap.add_argument('--ma-nguoi', required=True, help='W01, W02... (người đăng ký) hoặc U01... (người lạ); không dùng tên thật')
    ap.add_argument('--dieu-kien', required=True, choices=CONDITIONS)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--thu-muc', default=str(ROOT / 'data/raw/webcam'))
    args = ap.parse_args()

    base = Path(args.thu_muc)
    target = base / args.phien / args.ma_nguoi
    target.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem.rsplit('_', 1)[1]) for p in target.glob(f'{args.dieu_kien}_*.jpg') if p.stem.rsplit('_', 1)[1].isdigit()]
    count = max(existing, default=0)
    log_path = base / 'nhat_ky_thu_anh.csv'
    new_log = not log_path.exists()

    pipe = FacePipeline('B0')   # chỉ dùng detector để báo trạng thái khuôn mặt
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f'Không mở được webcam {args.camera}')
    viewer = Viewer('Thu ảnh webcam (Space: chụp, q: thoát)')
    message = ''
    with log_path.open('a', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(['thoi_gian', 'phien', 'ma_nguoi', 'dieu_kien', 'file', 'rong', 'cao', 'trang_thai_detector'])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            status, _, meta = pipe.prepare_face(image)
            shown = image.transpose(Image.FLIP_LEFT_RIGHT)      # chỉ lật để người chụp dễ nhìn
            draw = ImageDraw.Draw(shown)
            w, h = shown.size
            color = (46, 160, 67) if status == 'single_face' else (208, 59, 59)
            for x1, y1, x2, y2 in meta['boxes']:
                draw.rectangle([w - x2, y1, w - x1, y2], outline=color, width=3)
            f = font(max(14, h // 30))
            lines = [f'{args.phien} · {args.ma_nguoi} · {args.dieu_kien} · đã chụp {count}',
                     'Đúng 1 mặt' if status == 'single_face' else f'Detector: {status} ({meta["n_faces"]} mặt)', message]
            draw.rectangle([0, 0, w, 3 * (f.size + 6) + 6], fill=(20, 20, 20))
            for i, line in enumerate(lines):
                draw.text((8, 4 + i * (f.size + 6)), line, fill=(255, 255, 255) if i != 1 else color, font=f)
            keys = viewer.show(shown)
            if any(k in ('q', 'Q', 'Escape') for k in keys):
                break
            if 'space' in keys:
                if args.phien == 's1_dang_ky' and status != 'single_face':
                    message = 'Chưa lưu: ảnh đăng ký phải thấy đúng một mặt'
                    continue
                count += 1
                path = target / f'{args.dieu_kien}_{count}.jpg'
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                writer.writerow([datetime.now().isoformat(timespec='seconds'), args.phien, args.ma_nguoi, args.dieu_kien,
                                 path.relative_to(base).as_posix(), frame.shape[1], frame.shape[0], status])
                handle.flush()
                message = f'Đã lưu {path.name}'
    cap.release()
    viewer.close()
    print(f'{args.phien}/{args.ma_nguoi}: có {count} ảnh điều kiện "{args.dieu_kien}" trong {target}')


if __name__ == '__main__':
    main()
