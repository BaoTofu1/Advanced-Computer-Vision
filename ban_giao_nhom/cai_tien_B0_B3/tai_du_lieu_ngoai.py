"""Tải LFW và model ArcFace về data/external/ (không có trên GitHub vì dung lượng và giấy phép).

Chạy từ thư mục gốc repo:  python ban_giao_nhom/cai_tien_B0_B3/tai_du_lieu_ngoai.py
Cần thêm gói pyarrow để đọc file parquet của LFW:  pip install pyarrow

- LFW (Huang et al., 2007): bản sao trên Hugging Face `logasja/lfw` (13.233 ảnh, 5.749 người), vì
  nguồn chính thức (figshare, vis-www.cs.umass.edu) không tải được ngày 20/11/2026.
- ArcFace R50 `w600k_r50.onnx` trong gói buffalo_l của InsightFace (chỉ dùng nghiên cứu phi thương mại).
"""
from pathlib import Path
import hashlib
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'data/external'
LFW_URL = 'https://huggingface.co/datasets/logasja/lfw/resolve/main/data/train-00000-of-00001.parquet'
ARCFACE_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
ARCFACE_SHA256 = '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print('Tải', url)
        urllib.request.urlretrieve(url, target)
    return target


def lfw():
    out = EXT / 'lfw/images'
    if out.is_dir() and sum(1 for _ in out.glob('*/*.jpg')) == 13233:
        print('LFW đã có:', out); return
    import json
    import pyarrow.parquet as pq
    table = pq.read_table(download(LFW_URL, EXT / 'lfw/lfw_train.parquet'))
    names = json.loads(table.schema.metadata[b'huggingface'])['info']['features']['label']['names']
    counts = {}
    for row in table.to_pylist():
        name = names[row['label']]
        counts[name] = counts.get(name, 0) + 1
        path = out / name / f'{name}_{counts[name]:04d}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(row['image']['bytes'])
    print(f'LFW: {sum(counts.values())} ảnh, {len(counts)} người ->', out)


def arcface():
    target = EXT / 'arcface/w600k_r50.onnx'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == ARCFACE_SHA256:
        print('ArcFace đã có:', target); return
    archive = download(ARCFACE_URL, EXT / 'arcface/buffalo_l.zip')
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('w600k_r50.onnx'))
        data = z.read(name)
    assert hashlib.sha256(data).hexdigest() == ARCFACE_SHA256, 'File ArcFace khác bản nhóm đã dùng'
    target.write_bytes(data); archive.unlink()
    print('ArcFace ->', target)


if __name__ == '__main__':
    lfw()
    arcface()
