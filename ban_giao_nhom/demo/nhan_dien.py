"""Pipeline nhận diện dùng chung cho demo webcam, đo tốc độ và đánh giá ảnh webcam.

Khung hình -> MTCNN tìm mặt -> cắt 160x160 -> encoder (B0/B1/B2) -> so cosine với gallery.

    from nhan_dien import FacePipeline, Gallery
    pipe = FacePipeline('B0')
    gallery, report = Gallery.from_folder(pipe, 'data/raw/webcam/s1_dang_ky')
    status, crop, meta = pipe.prepare_face(anh_pil)
    if status == 'single_face':
        name, score, best = gallery.predict(pipe.encode([crop])[0], threshold=0.69)

Quy tắc giống các bước đánh giá: khung hình phải có đúng một mặt; không thấy mặt hoặc thấy
nhiều mặt thì trả trạng thái, không tự chọn một mặt và không trả embedding.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'huan_luyen'))
from tai_model import load_encoder  # noqa: E402

# Cấu hình MTCNN của bước 4 và bước 7.
MTCNN_CONFIG = {'image_size': 160, 'margin': 0, 'min_face_size': 20, 'thresholds': [0.6, 0.7, 0.7], 'factor': 0.709}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
UNKNOWN = 'Unknown'


def open_rgb(path):
    """Đọc ảnh từ file: xoay theo EXIF, nền trắng nếu có alpha, trả PIL RGB."""
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source)
        if 'A' in im.getbands():
            rgba = im.convert('RGBA')
            im = Image.alpha_composite(Image.new('RGBA', rgba.size, (255, 255, 255, 255)), rgba)
        return im.convert('RGB')


def face_status(boxes):
    if boxes is None:
        return 'no_face'
    try:
        boxes = np.asarray(boxes, dtype=np.float64)  # thư viện trả mảng dtype=object
    except (TypeError, ValueError):
        return 'invalid_boxes'
    if len(boxes) == 0:
        return 'no_face'
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        return 'invalid_boxes'
    return 'single_face' if len(boxes) == 1 else 'multiple_faces'


class FacePipeline:
    def __init__(self, model='B0', device=None, min_face_size=None):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        config = dict(MTCNN_CONFIG)
        if min_face_size:
            config['min_face_size'] = int(min_face_size)
        self.mtcnn_config = config
        self.mtcnn = MTCNN(keep_all=True, post_process=True, device=self.device, **config).eval()
        self._encoders = {}
        self.set_model(model)

    def set_model(self, name):
        if name not in self._encoders:
            self._encoders[name] = load_encoder(name, device=self.device)
        self.model_name = name
        self.encoder = self._encoders[name]

    @torch.inference_mode()
    def prepare_face(self, image):
        """PIL RGB -> (status, crop uint8 160x160x3 hoặc None, meta)."""
        boxes, probs = self.mtcnn.detect(image)
        status = face_status(boxes)
        meta = {'n_faces': 0 if boxes is None else len(boxes), 'boxes': [] if boxes is None else np.asarray(boxes, dtype=np.float32).tolist(),
                'prob': None}
        if status != 'single_face':
            return status, None, meta
        box = np.asarray(boxes, dtype=np.float32)
        meta['prob'] = float(probs[0])
        tensor = self.mtcnn.extract(image, box, save_path=None)[0].cpu()
        crop = (tensor * 128 + 127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return status, crop, meta

    @torch.inference_mode()
    def encode(self, crops, batch_size=64):
        """Danh sách crop uint8 160x160x3 -> float32 (N, 512), đã chuẩn hóa L2."""
        out = []
        for start in range(0, len(crops), batch_size):
            batch = np.stack(crops[start:start + batch_size])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
            x = ((x - 127.5) / 128.0).to(self.device)
            out.append(self.encoder(x).cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def read_gallery_folder(pipeline, folder):
    """<folder>/<mã người>/*.jpg -> ({mã người: [crop]}, báo cáo trạng thái từng ảnh).

    Crop không phụ thuộc model, nên đổi B0/B1/B2 chỉ cần tính lại embedding từ các crop này.
    """
    folder = Path(folder)
    crops_by_name, report = {}, []
    for person in (sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []):
        crops_by_name[person.name] = []
        for path in sorted(p for p in person.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            status, crop, _ = pipeline.prepare_face(open_rgb(path))
            report.append({'person': person.name, 'file': path.name, 'status': status})
            if crop is not None:
                crops_by_name[person.name].append(crop)
    return crops_by_name, report


class Gallery:
    """Template của từng người đã đăng ký: trung bình embedding (L2) rồi chuẩn hóa lại."""

    def __init__(self, names, templates, references):
        self.names = list(names)
        self.templates = np.asarray(templates, dtype=np.float32).reshape(len(self.names), -1)
        self.references = references          # {tên: crop uint8 dùng làm ảnh tham chiếu}

    def __len__(self):
        return len(self.names)

    @classmethod
    def from_crops(cls, pipeline, crops_by_name):
        names, templates, references = [], [], {}
        for name in sorted(crops_by_name):
            crops = crops_by_name[name]
            if not crops:
                continue
            t = pipeline.encode(crops).mean(axis=0)
            names.append(name); templates.append(t / np.linalg.norm(t)); references[name] = crops[0]
        return cls(names, templates, references)

    @classmethod
    def from_folder(cls, pipeline, folder):
        """Thư mục <folder>/<mã người>/*.jpg -> (Gallery, báo cáo từng ảnh)."""
        crops_by_name, report = read_gallery_folder(pipeline, folder)
        return cls.from_crops(pipeline, crops_by_name), report

    def predict(self, embedding, threshold):
        """-> (tên hoặc 'Unknown', cosine cao nhất, tên gần nhất)."""
        if not self.names:
            return UNKNOWN, float('nan'), None
        sim = self.templates @ np.asarray(embedding, dtype=np.float32)
        i = int(sim.argmax())
        score = float(sim[i])
        return (self.names[i] if score >= threshold else UNKNOWN), score, self.names[i]


class Viewer:
    """Cửa sổ hiển thị bằng tkinter.

    Môi trường dự án cài opencv-python-headless (không có cv2.imshow), nên phần hiển thị dùng
    tkinter có sẵn trong Python. `show()` vẽ một khung hình và trả về các phím vừa bấm, theo tên
    của tkinter: 'q', '1', 'plus', 'minus', 'space', 'Escape'...
    """

    def __init__(self, title, hidden=False):
        import tkinter as tk
        from PIL import ImageTk
        self._photo_class = ImageTk.PhotoImage
        self.root = tk.Tk()
        self.root.title(title)
        if hidden:                      # chỉ dùng khi kiểm thử tự động
            self.root.withdraw()
        self.label = tk.Label(self.root, borderwidth=0)
        self.label.pack()
        self.keys, self.closed = [], False
        self.root.bind('<Key>', lambda event: self.keys.append(event.keysym))
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.focus_force()

    def show(self, image):
        if self.closed:
            return ['Escape']
        self._photo = self._photo_class(image)       # giữ tham chiếu để ảnh không bị thu hồi
        self.label.configure(image=self._photo)
        self.root.update_idletasks()
        self.root.update()
        keys, self.keys = self.keys, []
        return keys

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
