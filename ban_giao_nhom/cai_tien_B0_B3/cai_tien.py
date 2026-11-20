"""Hàm dùng chung cho chuỗi cải tiến B0 → B3 (notebook 01–03 và app demo).

- `FaceAligner`: MTCNN tìm mặt, chọn mặt gần tâm ảnh nhất, cắt 160×160 cho FaceNet và
  căn 112×112 theo 5 mốc mặt cho ArcFace. Không thấy mặt thì dùng nguyên ảnh và mốc chuẩn.
- `add_mask`, `add_sunglasses`: vẽ khẩu trang / kính râm giả theo 5 mốc mặt, theo ý tưởng
  MaskTheFace (Anwar & Raychowdhury, 2020) nhưng dùng mốc của MTCNN thay cho dlib.
- `ArcFace`: chạy model ArcFace R50 (InsightFace buffalo_l, w600k_r50.onnx) bằng onnxruntime.
- `templates`, `best_match`: gallery nhiều template (ảnh rõ, ảnh đeo khẩu trang giả, ảnh đeo kính giả).
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from facenet_pytorch import MTCNN

MTCNN_CONFIG = {'image_size': 160, 'margin': 0, 'min_face_size': 20, 'thresholds': [0.6, 0.7, 0.7], 'factor': 0.709}
# Vị trí chuẩn của 5 mốc (mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải) trong ảnh 112×112 của ArcFace.
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
MASK_STYLES = {
    'surgical_blue': (150, 192, 228), 'surgical_white': (232, 233, 230), 'surgical_green': (158, 204, 184),
    'cloth_black': (34, 34, 38), 'cloth_gray': (118, 120, 126), 'n95_white': (238, 238, 234), 'cloth_pattern': (176, 64, 72),
}
GLASSES_STYLES = {'dark': (22, 22, 28), 'brown': (58, 40, 30), 'mirror': (70, 82, 104)}


class FaceAligner:
    """Tìm mặt bằng MTCNN (cấu hình bước 4/7) và chuẩn bị ảnh cho FaceNet và ArcFace."""

    def __init__(self, device=None, min_face_size=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        config = dict(MTCNN_CONFIG)
        if min_face_size:
            config['min_face_size'] = int(min_face_size)
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=device, **config).eval()

    @torch.inference_mode()
    def detect(self, image):
        """PIL RGB -> {status, box, landmarks (5×2, tọa độ ảnh), n_faces}.

        status: 'one_face' (thấy đúng 1 mặt), 'center_face' (nhiều mặt, chọn mặt gần tâm ảnh nhất),
        'fallback' (không thấy mặt: dùng nguyên ảnh, mốc mặt chuẩn).
        """
        w, h = image.size
        boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return {'status': 'fallback', 'box': np.array([0, 0, w, h], np.float32),
                    'landmarks': ARCFACE_TEMPLATE * np.array([w / 112, h / 112], np.float32), 'n_faces': 0}
        boxes = np.asarray(boxes, dtype=np.float32); points = np.asarray(points, dtype=np.float32)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        i = int(np.argmin(((centers - [w / 2, h / 2]) ** 2).sum(axis=1)))
        return {'status': 'one_face' if len(boxes) == 1 else 'center_face', 'box': boxes[i],
                'landmarks': points[i], 'n_faces': len(boxes)}

    @staticmethod
    def crop160(image, det):
        """Cắt theo khung mặt (margin 0) rồi resize bilinear 160×160, như facenet-pytorch. -> (uint8, mốc trong ảnh cắt)."""
        w, h = image.size
        x1, y1, x2, y2 = det['box']
        box = [int(max(x1, 0)), int(max(y1, 0)), int(min(x2, w)), int(min(y2, h))]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            box = [0, 0, w, h]
        crop = image.crop(box).resize((160, 160), Image.Resampling.BILINEAR)
        scale = np.array([160 / (box[2] - box[0]), 160 / (box[3] - box[1])], np.float32)
        return np.asarray(crop, dtype=np.uint8), (det['landmarks'] - np.array(box[:2], np.float32)) * scale

    @staticmethod
    def align112(image, landmarks):
        """Căn mặt theo 5 mốc về ảnh 112×112 (phép biến đổi tương tự, như InsightFace)."""
        matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            matrix = cv2.getAffineTransform(np.asarray(landmarks[:3], np.float32), ARCFACE_TEMPLATE[:3])
        return cv2.warpAffine(np.asarray(image, dtype=np.uint8), matrix, (112, 112), borderValue=0)


def _face_frame(landmarks):
    """Hệ tọa độ theo mặt: gốc ở giữa hai mắt, trục x theo đường nối hai mắt, đơn vị = khoảng cách hai mắt."""
    lm = np.asarray(landmarks, np.float32)
    le, re = lm[0], lm[1]
    center = (le + re) / 2
    ex = re - le
    e = float(np.linalg.norm(ex)) or 1.0
    ex = ex / e
    ey = np.array([-ex[1], ex[0]], np.float32)
    if ey[1] < 0:
        ey = -ey
    nose_b = float(np.dot(lm[2] - center, ey)) / e
    mouth_b = float(np.dot((lm[3] + lm[4]) / 2 - center, ey)) / e
    to_image = lambda pts: [tuple(center + e * (a * ex + b * ey)) for a, b in pts]
    return to_image, e, max(nose_b, 0.35), max(mouth_b, nose_b + 0.25)


def _shade(layer, rng, strength=0.10):
    """Thêm độ sáng tối nhẹ và nhiễu để vùng che bớt phẳng."""
    arr = np.asarray(layer, dtype=np.float32)
    h = arr.shape[0]
    grad = np.linspace(1 + strength, 1 - strength, h, dtype=np.float32)[:, None, None]
    noise = rng.normal(0, 4, size=arr.shape[:2] + (1,)).astype(np.float32)
    arr[..., :3] = np.clip(arr[..., :3] * grad + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), 'RGBA')


def add_mask(image, landmarks, style=None, rng=None):
    """Vẽ khẩu trang giả theo 5 mốc mặt. image: PIL hoặc uint8 RGB -> uint8 RGB cùng kích thước."""
    rng = rng if rng is not None else np.random.default_rng(0)
    base = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGBA')
    style = style or str(rng.choice(list(MASK_STYLES)))
    color = MASK_STYLES[style]
    to_image, e, nose_b, mouth_b = _face_frame(landmarks)
    top = nose_b * 0.38
    chin = mouth_b + max(0.68, (mouth_b - nose_b) * 1.45)
    outline = [(0, top - 0.06), (0.45, top - 0.01), (0.88, top + 0.12), (1.12, nose_b + 0.12), (1.08, mouth_b + 0.06),
               (0.9, mouth_b + 0.4), (0.5, chin - 0.06), (0, chin), (-0.5, chin - 0.06), (-0.9, mouth_b + 0.4),
               (-1.08, mouth_b + 0.06), (-1.12, nose_b + 0.12), (-0.88, top + 0.12), (-0.45, top - 0.01)]
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(to_image(outline), fill=color + (255,))
    line_w = max(1, int(round(e * 0.035)))
    darker = tuple(int(c * 0.78) for c in color) + (255,)
    if style.startswith('surgical'):
        for b in [top + 0.28 * (chin - top), top + 0.45 * (chin - top), top + 0.62 * (chin - top)]:
            draw.line(to_image([(-1.0, b), (1.0, b)]), fill=darker, width=line_w)
    elif style == 'n95_white':
        draw.line(to_image([(0, top - 0.03), (0, chin - 0.02)]), fill=darker, width=line_w)
    elif style == 'cloth_pattern':
        stripe = tuple(min(255, int(c * 1.35) + 30) for c in color) + (255,)
        for k, b in enumerate(np.linspace(top + 0.1, chin - 0.1, 7)):
            if k % 2 == 0:
                draw.line(to_image([(-1.05, b), (1.05, b)]), fill=stripe, width=line_w * 2)
        draw.polygon(to_image(outline), outline=darker)
    loop = (225, 225, 225, 255) if sum(color) < 300 else darker
    for side in (-1, 1):
        draw.line(to_image([(side * 1.1, nose_b + 0.05), (side * 1.22, top - 0.25)]), fill=loop, width=line_w)
        draw.line(to_image([(side * 0.98, mouth_b + 0.3), (side * 1.22, mouth_b + 0.42)]), fill=loop, width=line_w)
    alpha = layer.getchannel('A')
    layer = _shade(layer, rng)
    layer.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=max(0.5, e * 0.012))))
    return np.asarray(Image.alpha_composite(base, layer).convert('RGB'), dtype=np.uint8)


def add_sunglasses(image, landmarks, style=None, rng=None):
    """Vẽ kính râm giả theo vị trí hai mắt."""
    rng = rng if rng is not None else np.random.default_rng(0)
    base = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGBA')
    style = style or str(rng.choice(list(GLASSES_STYLES)))
    color = GLASSES_STYLES[style]
    to_image, e, _, _ = _face_frame(landmarks)
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ts = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    for cx in (-0.5, 0.5):
        lens = [(cx * 1.04 + 0.44 * np.cos(t), 0.03 + 0.31 * np.sin(t) + (0.06 if np.sin(t) > 0 else 0)) for t in ts]
        draw.polygon(to_image(lens), fill=color + (245,), outline=(10, 10, 10, 255))
        glare = [(cx * 1.04 - 0.2 + 0.08 * np.cos(t), -0.12 + 0.045 * np.sin(t)) for t in ts]
        draw.polygon(to_image(glare), fill=(255, 255, 255, 70))
    frame_w = max(1, int(round(e * 0.05)))
    draw.line(to_image([(-0.1, -0.04), (0.1, -0.04)]), fill=(15, 15, 15, 255), width=frame_w)
    for side in (-1, 1):
        draw.line(to_image([(side * 0.96, -0.04), (side * 1.2, -0.1)]), fill=(15, 15, 15, 255), width=frame_w)
    layer.putalpha(layer.getchannel('A').filter(ImageFilter.GaussianBlur(radius=max(0.4, e * 0.008))))
    return np.asarray(Image.alpha_composite(base, layer).convert('RGB'), dtype=np.uint8)


def add_occlusion(image, landmarks, kind, rng):
    """kind: 'clean' | 'mask' | 'sunglasses'."""
    if kind == 'mask':
        return add_mask(image, landmarks, rng=rng)
    if kind == 'sunglasses':
        return add_sunglasses(image, landmarks, rng=rng)
    return np.asarray(image, dtype=np.uint8)


class ArcFace:
    """ArcFace R50 (InsightFace buffalo_l, train trên WebFace600K) chạy bằng onnxruntime."""

    def __init__(self, model_path):
        import onnxruntime as ort
        providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input = self.session.get_inputs()[0].name

    def embed(self, aligned, batch=64):
        """Danh sách ảnh RGB uint8 112×112 -> float32 (N, 512), chuẩn hóa L2."""
        out = []
        for s in range(0, len(aligned), batch):
            x = np.stack(aligned[s:s + batch]).astype(np.float32)
            x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
            e = self.session.run(None, {self.input: x})[0]
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 512), np.float32)


@torch.inference_mode()
def facenet_embed(model, crops, batch=256):
    """Danh sách crop uint8 160×160 -> float32 (N, 512); chuẩn hóa (x-127.5)/128 đúng một lần."""
    device = next(model.parameters()).device
    out = []
    for s in range(0, len(crops), batch):
        x = torch.from_numpy(np.stack(crops[s:s + batch])).permute(0, 3, 1, 2).float()
        out.append(model(((x - 127.5) / 128.0).to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def templates(emb, persons):
    """Template mỗi người = trung bình embedding (L2) rồi chuẩn hóa lại. -> (tên, mảng)."""
    names = sorted(set(persons))
    persons = np.asarray(persons)
    t = np.stack([emb[persons == p].mean(axis=0) for p in names])
    return names, (t / np.linalg.norm(t, axis=1, keepdims=True)).astype(np.float32)


def best_match(query_emb, template_sets):
    """template_sets: danh sách mảng (P, 512) cùng thứ tự người (rõ, khẩu trang, kính...).
    Điểm của mỗi người = cosine lớn nhất qua các template. -> (chỉ số người, điểm)."""
    sim = np.max(np.stack([query_emb @ t.T for t in template_sets]), axis=0)
    idx = sim.argmax(axis=1)
    return idx, sim[np.arange(len(sim)), idx]


KINDS = ['clean', 'mask', 'sunglasses']


class LadderRecognizer:
    """Hệ thống nhận diện B0–B3 dùng cho app demo và notebook webcam.

    B0: FaceNet pretrained. B1: FaceNet fine-tune LFW + che giả. B2: B1 + gallery thêm template
    đeo khẩu trang/kính giả. B3: B2 ghép song song ArcFace (điểm = w·FaceNet + (1−w)·ArcFace).
    Ngưỡng và w đọc từ `khoa_cau_hinh_B0_B3.json` (chọn trên dev LFW), có thể ghi đè bằng
    `thresholds` (ví dụ ngưỡng chọn trên ảnh webcam phiên 2).
    """

    SYSTEMS = ['B0', 'B1', 'B2', 'B3']

    def __init__(self, system='B3', device=None, root=None, thresholds=None, aligner=None, min_face_size=None):
        import json
        self.here = Path(__file__).resolve().parent
        self.root = Path(root) if root else self.here.parents[1]
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lock = json.loads((self.here / 'khoa_cau_hinh_B0_B3.json').read_text(encoding='utf-8'))
        self.thresholds = {s: float(self.lock['systems'][s]['threshold']) for s in self.SYSTEMS}
        self.thresholds.update(thresholds or {})
        self.aligner = aligner or FaceAligner(self.device, min_face_size=min_face_size)
        self._facenet, self._arc, self._gallery = {}, None, {}
        self.set_system(system)

    # ---- model ----
    def set_system(self, system):
        import sys
        assert system in self.SYSTEMS, system
        fn = 'B0' if system == 'B0' else 'B1'
        if fn not in self._facenet:
            sys.path.insert(0, str(self.here.parent / 'huan_luyen'))
            from tai_model import load_encoder
            self._facenet[fn] = (load_encoder('B0', device=self.device) if fn == 'B0' else
                                 load_encoder('B1', device=self.device, checkpoint=self.here / 'checkpoints/B1_LFW_mask_encoder.pt'))
        if system == 'B3' and self._arc is None:
            self._arc = ArcFace(self.root / 'data/external/arcface/w600k_r50.onnx')
        self.system, self.fn_name = system, fn
        self.w = self.lock['systems'][system]['w_facenet']
        self._templates = None

    @property
    def threshold(self):
        return self.thresholds[self.system]

    def embed(self, items):
        """items: danh sách (crop160, aligned112) -> (FaceNet emb, ArcFace emb hoặc None)."""
        fn = facenet_embed(self._facenet[self.fn_name], [c for c, _ in items])
        arc = self._arc.embed([a for _, a in items]) if self.system == 'B3' else None
        return fn, arc

    # ---- gallery ----
    def prepare(self, image, det, kind='clean', rng=None):
        if kind != 'clean':
            image = Image.fromarray(add_occlusion(image, det['landmarks'], kind, rng or np.random.default_rng(0)))
        return FaceAligner.crop160(image, det)[0], FaceAligner.align112(image, det['landmarks'])

    def enroll(self, people):
        """people: {tên: [ảnh PIL]}. Ảnh không thấy mặt bị bỏ qua. -> báo cáo từng ảnh."""
        report = []
        self._gallery, self._templates = {}, None
        for name, images in people.items():
            for k, image in enumerate(images):
                det = self.aligner.detect(image)
                report.append({'person': name, 'index': k, 'status': det['status']})
                if det['status'] == 'fallback':
                    continue
                import hashlib
                seed = int(hashlib.sha256(f'{name}|{k}'.encode()).hexdigest()[:16], 16)
                items = {kind: self.prepare(image, det, kind, np.random.default_rng(seed + i)) for i, kind in enumerate(KINDS)}
                self._gallery.setdefault(name, []).append(items)
        return report

    def reference(self, name):
        return self._gallery[name][0]['clean'][0]

    @property
    def names(self):
        return sorted(self._gallery)

    def _build(self):
        kinds = ['clean'] if self.system in ('B0', 'B1') else KINDS
        sets = []
        for kind in kinds:
            items, owners = [], []
            for name in self.names:
                for entry in self._gallery[name]:
                    items.append(entry[kind]); owners.append(name)
            fn, arc = self.embed(items)
            sets.append((templates(fn, owners)[1], templates(arc, owners)[1] if arc is not None else None))
        self._templates = sets

    def scores(self, fn_emb, arc_emb=None):
        """-> ma trận điểm (N query × số người) theo hệ thống hiện tại."""
        if self._templates is None:
            self._build()
        if self.system == 'B3':
            return np.max(np.stack([self.w * (fn_emb @ f.T) + (1 - self.w) * (arc_emb @ a.T) for f, a in self._templates]), axis=0)
        return np.max(np.stack([fn_emb @ f.T for f, _ in self._templates]), axis=0)

    # ---- nhận diện một khung hình ----
    def identify(self, image):
        """-> dict(status, name, score, nearest, box, boxes, n_faces).

        status: 'recognized' | 'unknown' | 'no_face' | 'multiple_faces' | 'empty_gallery'.
        Khung hình phải có đúng một mặt (phạm vi đề tài); không tự chọn một mặt khi có nhiều người.
        """
        det = self.aligner.detect(image)
        out = {'n_faces': det['n_faces'], 'box': det['box'].tolist(), 'boxes': [], 'name': '', 'nearest': '', 'score': float('nan')}
        if det['status'] == 'fallback':
            return {**out, 'status': 'no_face', 'box': None}
        if det['status'] == 'center_face':
            boxes, _ = self.aligner.mtcnn.detect(image)
            return {**out, 'status': 'multiple_faces', 'boxes': np.asarray(boxes, np.float32).tolist()}
        if not self._gallery:
            return {**out, 'status': 'empty_gallery'}
        fn, arc = self.embed([self.prepare(image, det)])
        sim = self.scores(fn, arc)[0]
        i = int(sim.argmax())
        name = self.names[i]
        ok = sim[i] >= self.threshold
        return {**out, 'status': 'recognized' if ok else 'unknown', 'name': name if ok else 'Unknown',
                'nearest': name, 'score': float(sim[i])}
