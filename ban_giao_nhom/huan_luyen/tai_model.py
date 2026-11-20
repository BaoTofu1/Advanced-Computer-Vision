"""Nạp model nhận diện đã train (B0/B1/B2) để trích embedding, không cần train lại.

Ví dụ (chạy trong thư mục ban_giao_nhom/huan_luyen hoặc thêm thư mục này vào sys.path):

    from tai_model import load_encoder, embed
    model = load_encoder('B2')          # đọc checkpoints_full/B2_best_encoder.pt
    vectors = embed(model, [anh_mat])   # mảng (N, 512), mỗi dòng đã chuẩn hóa L2

`anh_mat` là ảnh khuôn mặt đã cắt (PIL.Image hoặc mảng RGB uint8). Ảnh được resize
bilinear về 160x160 rồi chuẩn hóa (x - 127.5) / 128 đúng một lần, giống lúc train.
Với webcam, cắt mặt bằng MTCNN trước rồi mới đưa vào `embed`.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1

HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / 'checkpoints_full'


def load_encoder(name='B2', device=None, checkpoint=None):
    """Trả về InceptionResnetV1 ở chế độ eval, output embedding 512 chiều.

    name: 'B0' (pretrained VGGFace2, không fine-tune), 'B1' hoặc 'B2'.
    checkpoint: đường dẫn file .pt khác nếu không dùng file mặc định.
    """
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if name == 'B0' and checkpoint is None:
        model = InceptionResnetV1(pretrained='vggface2', classify=False)
        model.checkpoint_info = {'branch': 'B0', 'source': 'facenet-pytorch vggface2'}
    else:
        path = Path(checkpoint) if checkpoint else CHECKPOINTS / f'{name}_best_encoder.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = InceptionResnetV1(pretrained=None, classify=False)
        model.load_state_dict(payload['encoder_state'], strict=True)
        model.checkpoint_info = {k: v for k, v in payload.items() if k != 'encoder_state'}
    return model.to(device).eval()


def to_rgb160(image):
    """PIL.Image hoặc mảng RGB uint8 -> mảng 160x160x3 uint8."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Cần mảng RGB uint8 dạng (H, W, 3)')
        if image.shape[:2] == (160, 160):
            return image
        image = Image.fromarray(image)
    image = image.convert('RGB')
    if image.size != (160, 160):
        image = image.resize((160, 160), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


@torch.inference_mode()
def embed(model, images, batch_size=256):
    """Danh sách ảnh -> embedding float32 (N, 512)."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(images), batch_size):
        batch = np.stack([to_rgb160(im) for im in images[start:start + batch_size]])
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
        x = ((x - 127.5) / 128.0).to(device)
        out.append(model(x).cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
