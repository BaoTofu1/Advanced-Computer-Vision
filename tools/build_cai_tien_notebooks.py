"""Dựng 3 notebook của chuỗi cải tiến B0 → B3 (ban_giao_nhom/cai_tien_B0_B3/)."""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]   # thư mục gốc của repo
HERE = ROOT / 'ban_giao_nhom/cai_tien_B0_B3'

SETUP = '''
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import os, sys, json, hashlib, time, gc, math, platform, inspect, warnings
warnings.filterwarnings('ignore', message='IProgress not found')
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Code
import torch


def find_root():
    options = [Path(os.environ['CV_PROJECT_ROOT'])] if os.environ.get('CV_PROJECT_ROOT') else []
    options += [Path.cwd(), *Path.cwd().parents]
    for p in options:
        if (p / 'ban_giao_nhom/cai_tien_B0_B3/cai_tien.py').is_file():
            return p.resolve()
    raise FileNotFoundError('Mở notebook trong thư mục dự án hoặc đặt CV_PROJECT_ROOT')


ROOT = find_root()
HERE = ROOT / 'ban_giao_nhom/cai_tien_B0_B3'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'ban_giao_nhom/huan_luyen'))
import cai_tien
from cai_tien import FaceAligner, ArcFace, add_occlusion, facenet_embed, templates, best_match
from tai_model import load_encoder
LFW_DIR = ROOT / 'data/external/lfw/images'
ARCFACE_PATH = ROOT / 'data/external/arcface/w600k_r50.onnx'
SPLIT_DIR = HERE / 'outputs_01_du_lieu'
SEED = 20261120
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
START = time.perf_counter()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def vn(x, d=1):
    return f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def pct(x, d=1):
    return vn(100 * x, d) + '%'


plt.rcParams.update({'font.family': 'Arial', 'figure.dpi': 110, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': .6})
COLORS = {'B0': '#8a8984', 'B1': '#2a78d6', 'B2': '#1baf7a', 'B3': '#eb6834'}
'''


def make(name, title_md, parts, out_dir):
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    code = lambda s: cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
    md(title_md)
    for kind, text in parts:
        (md if kind == 'md' else code)(text)
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3.11 (.venv đồ án)', 'language': 'python', 'name': 'cv-claude-full'},
        'language_info': {'name': 'python', 'version': '3.11.9'}})
    path = HERE / name
    same = False
    if path.exists():          # giữ output đã chạy nếu nội dung các ô không đổi
        old = nbf.read(path, as_version=4)
        same = [(c.cell_type, c.source) for c in old.cells] == [(c.cell_type, c.source) for c in nb.cells]
    if not same:
        nbf.write(nb, path)
    print(path.name, {'cells': len(cells), 'code_cells': sum(c.cell_type == 'code' for c in cells), 'giữ output': same})


OUT_HELPERS = '''
OUT = HERE / '{out}'; TAB, FIG, CACHE = OUT / 'tables', OUT / 'figures', OUT / 'cache'
for p in [OUT, TAB, FIG, CACHE]:
    p.mkdir(parents=True, exist_ok=True)


def save(d, name):
    d.to_csv(TAB / (name + '.csv'), index=False, encoding='utf-8-sig', float_format='%.17g'); return d


def finish(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(FIG / (name + '.' + ext), bbox_inches='tight', dpi=160)
    display(fig); plt.close(fig)
'''

# ----------------------------------------------------------------------------- 01
make('01_Du_lieu_LFW_va_che_gia.ipynb', '''
# B0 → B3 · 01. Dữ liệu mặt thật (LFW) và khẩu trang/kính giả

**Đồ án CV nâng cao · 20/11/2026**

Bước 7 cho thấy fine-tune trên ảnh tổng hợp (DigiFace) làm model kém trên ảnh người thật. Chuỗi cải tiến mới dựa trên bài báo của Anwar & Raychowdhury (2020), *Masked Face Recognition for Secure Authentication* (arXiv:2008.11104): dùng FaceNet, đeo khẩu trang giả lên **ảnh mặt thật** bằng công cụ MaskTheFace, rồi train trên ảnh đó. Bài báo cũng là nơi công bố bộ MFR2 mà nhóm dùng để test.

| | Thay đổi so với bước trước | Cơ sở |
|---|---|---|
| **B0** | FaceNet (InceptionResnetV1, VGGFace2) pretrained, ngưỡng chọn trên ảnh thật rõ mặt | Baseline của bài báo; FaceNet (Schroff 2015) |
| **B1** | **Đổi trọng số:** fine-tune nhẹ trên ảnh mặt thật đeo khẩu trang/kính giả | Cách làm của bài báo |
| **B2** | **Đổi ngưỡng và cách so khớp:** gallery thêm template đeo khẩu trang/kính giả, ngưỡng chọn trên ảnh thật có che | Bước 7: ngưỡng phải chọn đúng miền ảnh |
| **B3** | **Thêm model song song:** ghép điểm FaceNet (B2) với ArcFace | ArcFace (Deng 2019), InsightFace buffalo_l |

Notebook này chuẩn bị dữ liệu: LFW (Labeled Faces in the Wild, Huang 2007) làm ảnh mặt thật để train B1 và để chọn ngưỡng, sau khi loại những người trùng ROF/MFR2. ROF và MFR2 vẫn **chỉ để test**, đúng quyết định P3.
''', [
    ('code', SETUP + OUT_HELPERS.format(out='outputs_01_du_lieu')),
    ('md', '''
## 1. Kiểm kê LFW

LFW tải từ bản sao trên Hugging Face (`logasja/lfw`, file parquet 188 MB) vì hai nguồn chính thức (figshare, UMass) không tải được ngày 20/11/2026. Số ảnh và số người phải khớp bộ gốc: 13.233 ảnh, 5.749 người. Ảnh 250×250, người cần nhận nằm giữa ảnh.
'''),
    ('code', '''
files = sorted(LFW_DIR.glob('*/*.jpg'))
with ThreadPoolExecutor(max_workers=6) as pool:
    hashes = list(pool.map(sha, files))
L = pd.DataFrame({'image_id': ['LFW_' + f.stem for f in files], 'identity': [f.parent.name for f in files],
                  'path': [f.relative_to(ROOT).as_posix() for f in files], 'sha256': hashes})
assert len(L) == 13233 and L.identity.nunique() == 5749, 'Không khớp số liệu LFW gốc'
counts = L.groupby('identity').size()
fig, ax = plt.subplots(figsize=(8, 3.2))
bins = [1, 2, 3, 4, 6, 11, 21, 51, 600]
labels = ['1', '2', '3', '4–5', '6–10', '11–20', '21–50', '>50']
vals = pd.cut(counts, bins=bins, right=False, labels=labels).value_counts().reindex(labels)
ax.bar(labels, vals.values, color='#2a78d6', zorder=3); ax.bar_label(ax.containers[0], fontsize=8, color='#52514e')
ax.set(xlabel='Số ảnh của một người', ylabel='Số người', title='LFW: phần lớn người chỉ có 1 ảnh'); fig.tight_layout(); finish(fig, '01_lfw_images_per_person')
display(Markdown(f'**{vn(len(L), 0)} ảnh, {vn(L.identity.nunique(), 0)} người**; {vn((counts >= 4).sum(), 0)} người có từ 4 ảnh trở lên.'))
'''),
    ('md', '''
## 2. Loại người trùng với ROF và MFR2

LFW, ROF và MFR2 đều là ảnh người nổi tiếng nên có thể trùng người. Ai trùng tên (bỏ dấu cách, gạch dưới, chữ hoa) với một người trong ROF hoặc MFR2 bị loại khỏi mọi vai trò ở LFW, để model không được học hay chọn ngưỡng trên chính người sẽ test.
'''),
    ('code', '''
norm = lambda s: ''.join(ch for ch in s.lower() if ch.isalnum())
PROT = ROOT / 'ban_giao_nhom/protocol/notebook_outputs/tables'
ext = pd.read_csv(PROT / '07_external_image_roles_draft.csv', dtype=str, keep_default_na=False, usecols=['source', 'identity'])
test_names = {norm(x) for x in ext.identity}
L['norm'] = L.identity.map(norm)
overlap = sorted(set(L.loc[L.norm.isin(test_names), 'identity']))
removed = save(L[L.identity.isin(overlap)].groupby('identity').size().rename('ảnh').reset_index(), '01_removed_overlap_with_ROF_MFR2')
L = L[~L.identity.isin(overlap)].reset_index(drop=True)
display(removed.sort_values('ảnh', ascending=False).head(15))
display(Markdown(f'Loại **{len(overlap)} người** ({vn(removed["ảnh"].sum(), 0)} ảnh) trùng ROF/MFR2. Còn {vn(len(L), 0)} ảnh, {vn(L.identity.nunique(), 0)} người.'))
'''),
    ('md', '''
## 3. Tìm mặt và mốc mặt

MTCNN với cấu hình của bước 4/7. Ảnh LFW thường có người khác ở nền, nên khi thấy nhiều mặt thì chọn mặt gần tâm ảnh nhất (người cần nhận luôn ở giữa). Không thấy mặt thì dùng nguyên ảnh với mốc mặt chuẩn. Quy tắc này dùng chung cho LFW, ROF, MFR2 và app demo.
'''),
    ('code', '''
display(Code(inspect.getsource(FaceAligner.detect), language='python'))
cache = CACHE / 'lfw_detections.npz'
if cache.exists():
    with np.load(cache, allow_pickle=False) as z:
        assert z['image_id'].tolist() == L.image_id.tolist()
        BOX, LM, STATUS, NF = z['box'], z['lm'], z['status'], z['n_faces']
else:
    aligner = FaceAligner(device=DEVICE)
    dets = [aligner.detect(Image.open(ROOT / p).convert('RGB')) for p in L.path]
    BOX = np.stack([d['box'] for d in dets]); LM = np.stack([d['landmarks'] for d in dets])
    STATUS = np.array([d['status'] for d in dets], dtype='U'); NF = np.array([d['n_faces'] for d in dets])
    np.savez(cache, image_id=L.image_id.to_numpy(dtype='U'), box=BOX, lm=LM, status=STATUS, n_faces=NF)
L['status'] = STATUS
display(L.status.value_counts().rename('ảnh').to_frame())
'''),
    ('md', '''
## 4. Chia train / dev theo người

- **Train B1:** 70% số người có ít nhất 4 ảnh, tối đa 40 ảnh mỗi người (George W. Bush có 530 ảnh, không để một người chiếm phần lớn).
- **Dev người quen:** 30% còn lại. Mỗi người 2 ảnh làm gallery, tối đa 8 ảnh làm query.
- **Dev người lạ:** 600 người chỉ có 1–3 ảnh (không bao giờ vào gallery), mỗi người 1 ảnh.

Mỗi query dev có ba biến thể: rõ mặt, đeo khẩu trang giả, đeo kính giả (kiểu che chọn theo seed của ảnh). Dev dùng để chọn epoch cho B1 và chọn ngưỡng/trọng số ghép cho B0–B3; ROF/MFR2 không tham gia.
'''),
    ('code', '''
order = lambda xs, ctx: sorted(xs, key=lambda x: seed_of(SEED, ctx, x))
counts = L.groupby('identity').size()
multi = order(counts[counts >= 4].index, 'multi')
n_dev = round(0.3 * len(multi))
dev_known, train_ids = set(multi[:n_dev]), set(multi[n_dev:])
few = order(counts[counts <= 3].index, 'few')[:600]
role = {}
for ident, g in L.groupby('identity'):
    ids = order(g.image_id, 'img')
    if ident in train_ids:
        role.update({x: 'train' for x in ids[:40]})
    elif ident in dev_known:
        role.update({x: 'dev_gallery' for x in ids[:2]}); role.update({x: 'dev_known_query' for x in ids[2:10]})
    elif ident in few:
        role[ids[0]] = 'dev_unknown_query'
L['role'] = L.image_id.map(role).fillna('unused')
SPLIT = L[L.role.ne('unused')].reset_index(drop=True)
kinds = ['clean', 'mask', 'sunglasses']
SPLIT['occlusion_seed'] = [str(seed_of(SEED, 'occ', x)) for x in SPLIT.image_id]
summary = SPLIT.groupby('role').agg(ảnh=('image_id', 'size'), người=('identity', 'nunique')).reset_index()
assert not set(SPLIT[SPLIT.role.eq('train')].identity) & set(SPLIT[SPLIT.role.ne('train')].identity)
assert not set(SPLIT[SPLIT.role.eq('dev_unknown_query')].identity) & set(SPLIT[SPLIT.role.eq('dev_gallery')].identity)
cols = ['image_id', 'identity', 'path', 'sha256', 'role', 'status', 'occlusion_seed']
SPLIT[cols].to_csv(OUT / 'lfw_split.csv', index=False, encoding='utf-8-sig')
idx = L.set_index('image_id').index.get_indexer(SPLIT.image_id)
np.savez(OUT / 'lfw_split_detections.npz', image_id=SPLIT.image_id.to_numpy(dtype='U'), box=BOX[idx], lm=LM[idx], status=STATUS[idx])
save(summary, '02_split_summary'); display(summary)
'''),
    ('md', '''
## 5. Khẩu trang và kính râm giả

Vẽ theo 5 mốc mặt, trong hệ tọa độ gắn với đường nối hai mắt nên xoay theo mặt nghiêng. Khẩu trang có 7 kiểu (y tế xanh/trắng/xanh lá, vải đen/xám/họa tiết, N95), phủ từ sống mũi xuống cằm; kính râm 3 kiểu. Đây là phiên bản đơn giản của MaskTheFace: MaskTheFace dán ảnh mẫu khẩu trang theo 68 mốc của dlib, còn ở đây vẽ đa giác theo 5 mốc của MTCNN để không phải cài dlib.
'''),
    ('code', '''
display(Code(inspect.getsource(cai_tien.add_mask), language='python'))
sample = SPLIT[SPLIT.role.eq('dev_gallery')].drop_duplicates('identity').head(6)
lm_by = dict(zip(SPLIT.image_id, np.load(OUT / 'lfw_split_detections.npz')['lm']))
fig, axes = plt.subplots(3, 6, figsize=(14, 7.2))
for j, r in enumerate(sample.itertuples()):
    im = Image.open(ROOT / r.path).convert('RGB')
    for i, kind in enumerate(kinds):
        axes[i, j].imshow(add_occlusion(im, lm_by[r.image_id], kind, np.random.default_rng(seed_of(SEED, kind, r.image_id))))
        axes[i, j].set_title(f'{r.identity[:18]} · {kind}', fontsize=8)
for ax in axes.flat:
    ax.axis('off')
fig.tight_layout(); finish(fig, '02_synthetic_occlusions')
json.dump({'checks': 'PASS', 'lfw_images': 13233, 'removed_overlap_identities': len(overlap), 'split': summary.to_dict('records'),
           'detections': L.status.value_counts().to_dict(), 'seed': SEED, 'executed_at_utc': datetime.now(timezone.utc).isoformat(),
           'elapsed_seconds': time.perf_counter() - START}, open(OUT / 'execution_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
'''),
], 'outputs_01_du_lieu')

# ----------------------------------------------------------------------------- 02
make('02_Train_B1_LFW_khau_trang.ipynb', '''
# B0 → B3 · 02. B1: fine-tune FaceNet trên ảnh mặt thật đeo khẩu trang giả

**Đồ án CV nâng cao · 20/11/2026**

Theo bài báo của Anwar & Raychowdhury (2020): train FaceNet trên ảnh mặt thật có khẩu trang giả. Bài báo train lại từ đầu trên 360.000 ảnh (VGGFace2-mini), còn nhóm chỉ có vài nghìn ảnh LFW, nên ở đây **fine-tune nhẹ**: giữ gần hết mạng, chỉ mở các khối cuối, BatchNorm giữ nguyên thống kê, learning rate nhỏ. Rút kinh nghiệm từ bước 6–7: fine-tune toàn bộ mạng trên dữ liệu khác miền làm model quên đặc trưng của mặt thật.

Mỗi epoch chấm trên dev LFW (người chưa thấy lúc train) với ba biến thể query: rõ mặt, khẩu trang giả, kính giả. Epoch được chọn theo Rank-1 trung bình của ba biến thể.
''', [
    ('code', SETUP + OUT_HELPERS.format(out='outputs_02_train_B1') + '''
from torch import nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
CFG = {'epochs': 8, 'warmup_epochs': 1, 'batch_size': 32, 'lr_head': 1e-3, 'lr_encoder': 2e-5, 'weight_decay': 1e-4,
       'open_layers': ['repeat_3', 'block8', 'last_linear', 'last_bn'], 'bn_running_stats': 'frozen',
       'occlusion_probability': 0.5, 'mask_share': 0.7, 'hflip_probability': 0.5, 'seed': SEED,
       'selection': 'max mean dev Rank-1 over clean/mask/sunglasses; tie: earlier epoch'}
display(pd.DataFrame([{'mục': k, 'giá trị': str(v)} for k, v in CFG.items()]))
'''),
    ('md', '''
## 1. Ảnh train và dev

Ảnh train cắt sẵn 160×160 theo khung mặt (kèm mốc mặt trong ảnh cắt) để che giả và lật ngang ngay lúc train. Ảnh dev: che giả trên ảnh gốc rồi mới cắt, giống cách app xử lý.
'''),
    ('code', '''
SPLIT = pd.read_csv(SPLIT_DIR / 'lfw_split.csv', dtype=str, keep_default_na=False)
with np.load(SPLIT_DIR / 'lfw_split_detections.npz', allow_pickle=False) as z:
    assert z['image_id'].tolist() == SPLIT.image_id.tolist()
    DET = [{'box': b, 'landmarks': l} for b, l in zip(z['box'], z['lm'])]
for r in SPLIT.sample(64, random_state=0).itertuples():
    assert sha(ROOT / r.path) == r.sha256


def prepare(i, kind='clean'):
    r = SPLIT.iloc[i]; im = Image.open(ROOT / r.path).convert('RGB')
    if kind != 'clean':
        im = Image.fromarray(add_occlusion(im, DET[i]['landmarks'], kind, np.random.default_rng(int(r.occlusion_seed))))
    return FaceAligner.crop160(im, DET[i])


tr = np.flatnonzero(SPLIT.role.eq('train').to_numpy())
with ThreadPoolExecutor(max_workers=6) as pool:
    out = list(pool.map(prepare, tr))
TRAIN_X = np.stack([o[0] for o in out]); TRAIN_LM = np.stack([o[1] for o in out])
labels = {x: i for i, x in enumerate(sorted(SPLIT.iloc[tr].identity.unique()))}
TRAIN_Y = np.array([labels[x] for x in SPLIT.iloc[tr].identity], dtype=np.int64)
DEV = {}
dev_idx = np.flatnonzero(SPLIT.role.ne('train').to_numpy())
for kind in ['clean', 'mask', 'sunglasses']:
    with ThreadPoolExecutor(max_workers=6) as pool:
        DEV[kind] = np.stack([o[0] for o in pool.map(lambda i: prepare(i, kind), dev_idx)])
DEV_FRAME = SPLIT.iloc[dev_idx].reset_index(drop=True)
display(Markdown(f'Train: **{len(TRAIN_X)} ảnh, {len(labels)} người**. Dev: {len(DEV_FRAME)} ảnh × 3 biến thể.'))
'''),
    ('md', '''
## 2. Cách chấm trên dev

Gallery = 2 ảnh rõ mặt mỗi người quen (template trung bình). Query người quen và người lạ ở cả ba biến thể. Rank-1 không cần ngưỡng nên dùng để chọn epoch; DIR tại FPIR ≤ 5% được báo thêm, với ngưỡng chọn trên query rõ mặt (cách của B0/B1).
'''),
    ('code', '''
g_mask = DEV_FRAME.role.eq('dev_gallery').to_numpy()
q_mask = ~g_mask
Q = DEV_FRAME[q_mask].reset_index(drop=True)
known = Q.role.eq('dev_known_query').to_numpy()


def choose_threshold(score, correct, known):
    cand = np.r_[np.unique(score).astype(np.float64), np.inf]
    hit = np.sort(score[known & correct]); unk = np.sort(score[~known])
    DIR = (len(hit) - np.searchsorted(hit, cand, side='left')) / known.sum()
    FPIR = (len(unk) - np.searchsorted(unk, cand, side='left')) / (~known).sum()
    ok = FPIR <= .05
    return float(cand[ok & (DIR == DIR[ok].max())].max())


def dev_eval(model):
    model.classify = False; model.eval()
    emb = {k: facenet_embed(model, list(v)) for k, v in DEV.items()}
    names, T = templates(emb['clean'][g_mask], DEV_FRAME.identity[g_mask].to_numpy())
    res = {}
    for kind in DEV:
        idx, score = best_match(emb[kind][q_mask], [T])
        correct = known & (np.asarray(names)[idx] == Q.identity.to_numpy())
        res[kind] = (score.astype(np.float32), correct)
    t = choose_threshold(*res['clean'], known)
    row = {}
    for kind, (score, correct) in res.items():
        acc = score >= t
        row[f'Rank-1 {kind}'] = float(correct[known].mean()); row[f'DIR {kind}'] = float((acc & correct)[known].mean())
        row[f'FPIR {kind}'] = float(acc[~known].mean())
    row['Rank-1 trung bình'] = float(np.mean([row[f'Rank-1 {k}'] for k in DEV])); row['ngưỡng'] = t
    return row


b0 = load_encoder('B0', device=DEVICE)
B0_DEV = dev_eval(b0); del b0
display(pd.DataFrame([{'model': 'B0', **B0_DEV}]).round(4))
'''),
    ('md', '''
## 3. Train

Epoch 0 chỉ học classifier (encoder đóng băng). Các epoch sau mở `repeat_3`, `block8` và lớp chiếu cuối. Mỗi ảnh train có 50% khả năng bị che (70% khẩu trang, 30% kính), 50% bị lật ngang; seed theo (epoch, ảnh) nên chạy lại ra cùng kết quả.
'''),
    ('code', '''
torch.manual_seed(SEED); np.random.seed(SEED % 2**32)
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(labels)).to(DEVICE)
OPEN = tuple(p + '.' for p in CFG['open_layers'])
encoder_params = [p for n, p in model.named_parameters() if n.startswith(OPEN)]
optimizer = torch.optim.AdamW([{'params': model.logits.parameters(), 'lr': CFG['lr_head']},
                               {'params': encoder_params, 'lr': CFG['lr_encoder']}], weight_decay=CFG['weight_decay'])
FLIP = [1, 0, 2, 4, 3]


def augment(i, epoch):
    rng = np.random.default_rng(seed_of(SEED, 'train', epoch, i))
    x, lm = TRAIN_X[i], TRAIN_LM[i]
    if rng.random() < CFG['occlusion_probability']:
        x = add_occlusion(x, lm, 'mask' if rng.random() < CFG['mask_share'] else 'sunglasses', rng)
    if rng.random() < CFG['hflip_probability']:
        x = x[:, ::-1]
    return np.ascontiguousarray(x)


def set_mode(stage):
    model.classify = True; model.eval(); model.logits.train()
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith('logits.') or (stage == 'fine' and n.startswith(OPEN)))
    if stage == 'fine':
        for layer in CFG['open_layers']:
            getattr(model, layer).train()
        model.dropout.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


HIST, best = [{'epoch': -1, 'stage': 'B0', **B0_DEV}], None
CKPT = HERE / 'checkpoints'; CKPT.mkdir(exist_ok=True)
for epoch in range(CFG['epochs']):
    stage = 'warmup' if epoch < CFG['warmup_epochs'] else 'fine'; set_mode(stage)
    order = np.random.default_rng(seed_of(SEED, 'order', epoch)).permutation(len(TRAIN_X))
    loss_sum, hits, begin = 0., 0, time.perf_counter()
    for s in range(0, len(order), CFG['batch_size']):
        idx = order[s:s + CFG['batch_size']]
        x = torch.from_numpy(np.stack([augment(i, epoch) for i in idx])).permute(0, 3, 1, 2).float()
        x = ((x - 127.5) / 128).to(DEVICE); y = torch.from_numpy(TRAIN_Y[idx]).to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x); loss = F.cross_entropy(logits, y); loss.backward(); optimizer.step()
        loss_sum += float(loss) * len(idx); hits += int((logits.argmax(1) == y).sum())
    row = {'epoch': epoch, 'stage': stage, 'train_loss': loss_sum / len(order), 'train_acc': hits / len(order),
           'train_seconds': time.perf_counter() - begin, **dev_eval(model)}
    HIST.append(row)
    if stage == 'fine' and (best is None or row['Rank-1 trung bình'] > best['Rank-1 trung bình']):
        best = row
        torch.save({'encoder_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('logits.')},
                    'branch': 'B1_LFW_mask', 'epoch': epoch, 'metrics': {k: v for k, v in row.items() if isinstance(v, float) and k != 'train_seconds'},
                    'config': CFG, 'train_identities': len(labels), 'train_images': len(TRAIN_X)}, CKPT / 'B1_LFW_mask_encoder.pt')
    print(f"epoch {epoch} ({stage}): loss {row['train_loss']:.3f}, dev Rank-1 rõ/khẩu trang/kính "
          f"{row['Rank-1 clean']:.3f}/{row['Rank-1 mask']:.3f}/{row['Rank-1 sunglasses']:.3f}", flush=True)
HIST = save(pd.DataFrame(HIST), '01_training_history')
display(HIST.drop(columns=['stage']).round(4))
'''),
    ('code', '''
fig, ax = plt.subplots(figsize=(9, 3.8))
h = HIST[HIST.epoch >= 0]
for kind, color in [('clean', '#2a78d6'), ('mask', '#eb6834'), ('sunglasses', '#1baf7a')]:
    ax.plot(h.epoch, 100 * h[f'Rank-1 {kind}'], 'o-', color=color, lw=2, ms=7, label=f'B1 · {kind}')
    ax.axhline(100 * B0_DEV[f'Rank-1 {kind}'], color=color, ls=':', lw=1.5)
ax.axvline(best['epoch'], color='#52514e', lw=1, ls='--'); ax.text(best['epoch'], ax.get_ylim()[0], ' epoch chọn', fontsize=8, va='bottom')
ax.set(xlabel='Epoch (đường chấm = B0)', ylabel='Rank-1 trên dev LFW (%)', title='B1 theo epoch trên người chưa thấy lúc train')
ax.legend(frameon=False, fontsize=8, ncols=3); fig.tight_layout(); finish(fig, '01_dev_rank1_by_epoch')
diff = {k: best[f'Rank-1 {k}'] - B0_DEV[f'Rank-1 {k}'] for k in ['clean', 'mask', 'sunglasses']}
json.dump({'checks': 'PASS', 'config': CFG, 'best_epoch': best['epoch'], 'B0_dev': B0_DEV, 'B1_dev': best,
           'checkpoint_sha256': sha(CKPT / 'B1_LFW_mask_encoder.pt'), 'executed_at_utc': datetime.now(timezone.utc).isoformat(),
           'elapsed_seconds': time.perf_counter() - START}, open(OUT / 'execution_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
display(Markdown(f"""## 4. Kết luận

Epoch được chọn: **{best['epoch']}**. Trên dev LFW, Rank-1 của B1 so với B0: rõ mặt {pct(best['Rank-1 clean'])} ({100 * diff['clean']:+.1f} điểm), khẩu trang giả {pct(best['Rank-1 mask'])} ({100 * diff['mask']:+.1f}), kính giả {pct(best['Rank-1 sunglasses'])} ({100 * diff['sunglasses']:+.1f}). Encoder lưu ở `checkpoints/B1_LFW_mask_encoder.pt`. Dev là ảnh che giả; kết quả trên che thật nằm ở notebook 03.
""".replace('+', '+')))
'''),
], 'outputs_02_train_B1')

# ----------------------------------------------------------------------------- 03
make('03_Danh_gia_B0_B3.ipynb', '''
# B0 → B3 · 03. Chọn ngưỡng trên dev, khóa, rồi chấm ROF và MFR2

**Đồ án CV nâng cao · 20/11/2026**

| | Model | Gallery | Ngưỡng chọn trên |
|---|---|---|---|
| **B0** | FaceNet pretrained | template ảnh rõ | dev LFW rõ mặt |
| **B1** | FaceNet fine-tune LFW + che giả (notebook 02) | template ảnh rõ | dev LFW rõ mặt |
| **B2** | như B1 | thêm template đeo khẩu trang giả và kính giả; điểm = cosine lớn nhất | dev LFW có che (rõ + khẩu trang + kính) |
| **B3** | B2 ghép song song ArcFace: điểm = w·FaceNet + (1−w)·ArcFace | như B2, cho cả hai model | như B2; w chọn trên dev |

Mỗi bước chỉ thêm một thay đổi. Ngưỡng và w chọn hoàn toàn trên dev LFW, ghi vào `khoa_cau_hinh_B0_B3.json` trước khi notebook mở ảnh ROF/MFR2. Ảnh test xử lý giống app: MTCNN chọn mặt gần tâm, không thấy mặt thì dùng nguyên ảnh.
''', [
    ('code', SETUP + OUT_HELPERS.format(out='outputs_03_danh_gia') + '''
B1_PATH = HERE / 'checkpoints/B1_LFW_mask_encoder.pt'
KINDS = ['clean', 'mask', 'sunglasses']
SYSTEMS = ['B0', 'B1', 'B2', 'B3']
W_GRID = [0.25, 0.5, 0.75]
'''),
    ('md', '''
## 1. Embedding trên dev LFW

Với mỗi ảnh dev: ba biến thể (rõ, khẩu trang giả, kính giả), mỗi biến thể có crop 160×160 cho FaceNet và ảnh căn 112×112 cho ArcFace. Gallery cũng có ba biến thể (dùng cho B2/B3).
'''),
    ('code', '''
SPLIT = pd.read_csv(SPLIT_DIR / 'lfw_split.csv', dtype=str, keep_default_na=False)
with np.load(SPLIT_DIR / 'lfw_split_detections.npz', allow_pickle=False) as z:
    DET = {i: {'box': b, 'landmarks': l} for i, b, l in zip(z['image_id'], z['box'], z['lm'])}
DEVF = SPLIT[SPLIT.role.ne('train')].reset_index(drop=True)


def variants(frame, det, kinds=KINDS, seed_col='occlusion_seed'):
    """-> {kind: (crops160, aligned112)} cho mọi ảnh của frame."""
    out = {}
    for kind in kinds:
        def one(r):
            im = Image.open(ROOT / r.path).convert('RGB'); d = det[r.image_id]
            if kind != 'clean':
                im = Image.fromarray(add_occlusion(im, d['landmarks'], kind, np.random.default_rng(int(getattr(r, seed_col)))))
            return FaceAligner.crop160(im, d)[0], FaceAligner.align112(im, d['landmarks'])
        with ThreadPoolExecutor(max_workers=6) as pool:
            res = list(pool.map(one, frame.itertuples()))
        out[kind] = ([r[0] for r in res], [r[1] for r in res])
    return out


ENCODERS = {'B0': load_encoder('B0', device=DEVICE), 'B1': load_encoder('B1', device=DEVICE, checkpoint=B1_PATH)}
ARC = ArcFace(ARCFACE_PATH)


def embed_all(var):
    return {(m, k): facenet_embed(ENCODERS[m], var[k][0]) for m in ['B0', 'B1'] for k in var} | \\
           {('ARC', k): ARC.embed(var[k][1]) for k in var}


DEV_E = embed_all(variants(DEVF, DET))
display(Markdown(f'Dev: {len(DEVF)} ảnh × 3 biến thể × 3 model (B0, B1, ArcFace).'))
'''),
    ('md', '''
## 2. Bốn hệ thống và cách chấm chung

`system_scores` trả về, cho mỗi query, người gần nhất và điểm. B0/B1 so với template ảnh rõ; B2 lấy điểm lớn nhất qua ba template (rõ, khẩu trang giả, kính giả); B3 ghép điểm FaceNet và ArcFace theo trọng số w trên từng template rồi lấy lớn nhất.
'''),
    ('code', '''
def system_scores(system, E, gal, q_kind_rows, w=None):
    """E: embedding dict; gal: mask gallery trong frame; q_kind_rows: list (kind, boolean query mask)."""
    fn = 'B0' if system == 'B0' else 'B1'
    persons = FRAME_IDS[gal]
    def tsets(model):
        kinds = ['clean'] if system in ('B0', 'B1') else KINDS
        names, sets = None, []
        for k in kinds:
            names, t = templates(E[(model, k)][gal], persons); sets.append(t)
        return names, sets
    names, fn_sets = tsets(fn)
    out_idx, out_score = [], []
    for kind, qm in q_kind_rows:
        qf = E[(fn, kind)][qm]
        if system == 'B3':
            _, arc_sets = tsets('ARC'); qa = E[('ARC', kind)][qm]
            sim = np.max(np.stack([w * (qf @ a.T) + (1 - w) * (qa @ b.T) for a, b in zip(fn_sets, arc_sets)]), axis=0)
        else:
            sim = np.max(np.stack([qf @ t.T for t in fn_sets]), axis=0)
        i = sim.argmax(axis=1); out_idx.append(np.asarray(names)[i]); out_score.append(sim[np.arange(len(sim)), i])
    return np.concatenate(out_idx), np.concatenate(out_score).astype(np.float32)


def choose_threshold(score, correct, known):
    cand = np.r_[np.unique(score).astype(np.float64), np.inf]
    hit = np.sort(score[known & correct]); unk = np.sort(score[~known])
    DIR = (len(hit) - np.searchsorted(hit, cand, side='left')) / known.sum()
    FPIR = (len(unk) - np.searchsorted(unk, cand, side='left')) / (~known).sum()
    ok = FPIR <= .05
    return float(cand[ok & (DIR == DIR[ok].max())].max()), float(DIR[ok].max())


FRAME_IDS = DEVF.identity.to_numpy()
gal = DEVF.role.eq('dev_gallery').to_numpy(); qm = ~gal
Qd = DEVF[qm].reset_index(drop=True); known_d = Qd.role.eq('dev_known_query').to_numpy()


def dev_run(system, kinds, w=None):
    pred, score = system_scores(system, DEV_E, gal, [(k, qm) for k in kinds], w)
    truth = np.tile(Qd.identity.to_numpy(), len(kinds)); kn = np.tile(known_d, len(kinds))
    return score, kn & (pred == truth), kn, np.repeat(kinds, len(Qd))


CALIB = {}
for s in ['B0', 'B1']:
    CALIB[s] = {'kinds': ['clean'], 'w': None}
CALIB['B2'] = {'kinds': KINDS, 'w': None}
w_rows = []
for w in W_GRID:
    sc, cor, kn, _ = dev_run('B3', KINDS, w); t, d = choose_threshold(sc, cor, kn)
    w_rows.append({'w (FaceNet)': w, 'DIR dev có che @FPIR5%': d})
W_TABLE = save(pd.DataFrame(w_rows), '01_fusion_weight_on_dev')
best_w = float(W_TABLE.sort_values(['DIR dev có che @FPIR5%', 'w (FaceNet)'], ascending=[False, False]).iloc[0]['w (FaceNet)'])
CALIB['B3'] = {'kinds': KINDS, 'w': best_w}
for s in SYSTEMS:
    sc, cor, kn, _ = dev_run(s, CALIB[s]['kinds'], CALIB[s]['w'])
    CALIB[s]['threshold'] = choose_threshold(sc, cor, kn)[0]
dev_rows = []
for s in SYSTEMS:
    sc, cor, kn, kind = dev_run(s, KINDS, CALIB[s]['w'])
    acc = sc >= np.float32(CALIB[s]['threshold'])
    for k in KINDS:
        sel = kind == k
        dev_rows.append({'hệ thống': s, 'biến thể': k, 'Rank-1': float(cor[sel & kn].mean()), 'DIR': float((acc & cor)[sel & kn].mean()),
                         'FPIR': float(acc[sel & ~kn].mean())})
DEV_TABLE = save(pd.DataFrame(dev_rows), '02_dev_results')
display(W_TABLE.round(4))
display(DEV_TABLE.pivot(index='biến thể', columns='hệ thống', values=['Rank-1', 'DIR', 'FPIR']).mul(100).round(1))
'''),
    ('md', '''
## 3. Khóa cấu hình trước khi mở ảnh test
'''),
    ('code', '''
LOCK = {'version': 'B0_B3_v1', 'paper': 'Anwar & Raychowdhury 2020, arXiv:2008.11104',
        'systems': {s: {'threshold': CALIB[s]['threshold'], 'w_facenet': CALIB[s]['w'], 'calibration_variants': CALIB[s]['kinds']} for s in SYSTEMS},
        'models': {'B1_LFW_mask_encoder': sha(B1_PATH), 'arcface_w600k_r50': sha(ARCFACE_PATH), 'B0': 'facenet-pytorch vggface2'},
        'detection': 'MTCNN bước 4/7; nhiều mặt -> mặt gần tâm; không thấy -> nguyên ảnh + mốc chuẩn',
        'gallery_variants_B2_B3': KINDS, 'occlusion_seed': 'seed_of(SEED, "test-occ", image_id)', 'seed': SEED,
        'lfw_split_sha256': sha(SPLIT_DIR / 'lfw_split.csv'),
        'metrics': 'DIR/FPIR tại ngưỡng khóa, Rank-1; MFR2: AUC, EER, accuracy tại ngưỡng khóa; bootstrap theo người 2.000 lần'}
LOCK_PATH = HERE / 'khoa_cau_hinh_B0_B3.json'
FIRST = HERE / 'khoa_cau_hinh_B0_B3_lan1.json'
if FIRST.exists():
    # Lần khóa đầu (20/11) lưu checkpoint B1 có kèm thời gian train nên SHA đổi mỗi lần chạy, dù trọng số giống hệt.
    # Checkpoint giờ không còn thời gian train; ngưỡng, w và model ArcFace phải trùng đúng lần khóa đầu.
    first = json.loads(FIRST.read_text(encoding='utf-8'))
    assert first['systems'] == json.loads(json.dumps(LOCK['systems'])), 'Ngưỡng/w khác lần khóa đầu'
    assert first['models']['arcface_w600k_r50'] == LOCK['models']['arcface_w600k_r50']
    display(Markdown(f"Ngưỡng và w trùng đúng lần khóa đầu (`{FIRST.name}`, {first['locked_at_utc'][:16]} UTC)."))
if LOCK_PATH.exists():
    old = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    assert {k: v for k, v in old.items() if k != 'locked_at_utc'} == json.loads(json.dumps(LOCK, default=str)), 'Cấu hình khác file khóa'
    LOCK_STATUS = 'file khóa có sẵn, khớp'
else:
    LOCK_PATH.write_text(json.dumps({**LOCK, 'locked_at_utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    LOCK_STATUS = 'vừa ghi file khóa'
display(pd.DataFrame([{'hệ thống': s, **LOCK['systems'][s]} for s in SYSTEMS]))
display(Markdown(f'**{LOCK_STATUS}:** `{LOCK_PATH.name}`'))
'''),
    ('md', '''
## 4. ROF: ảnh che thật

Gallery 3 ảnh rõ mặt cho mỗi người quen (144 người), 36 người lạ. Với B2/B3, ba ảnh gallery được đeo thêm khẩu trang và kính giả để làm template phụ. Query là ảnh thật: rõ mặt, khẩu trang, kính râm.
'''),
    ('code', '''
PROT = ROOT / 'ban_giao_nhom/protocol/notebook_outputs/tables'
RO = pd.read_csv(PROT / '07_rof_image_roles_draft.csv', dtype=str, keep_default_na=False)
RO = RO[RO.role.isin(['gallery', 'known_query', 'unknown_query'])].sort_values('image_id').reset_index(drop=True)
RO['occlusion_seed'] = [str(seed_of(SEED, 'test-occ', x)) for x in RO.image_id]
aligner = FaceAligner(device=DEVICE)
cache = CACHE / 'rof_detections.npz'
if cache.exists():
    with np.load(cache, allow_pickle=False) as z:
        assert z['image_id'].tolist() == RO.image_id.tolist(); RDET = {i: {'box': b, 'landmarks': l, 'status': s} for i, b, l, s in zip(z['image_id'], z['box'], z['lm'], z['status'])}
else:
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert all(pool.map(lambda r: sha(ROOT / r.path) == r.sha256, RO.itertuples()))
    dets = [aligner.detect(Image.open(ROOT / p).convert('RGB')) for p in RO.path]
    np.savez(cache, image_id=RO.image_id.to_numpy(dtype='U'), box=np.stack([d['box'] for d in dets]), lm=np.stack([d['landmarks'] for d in dets]),
             status=np.array([d['status'] for d in dets], dtype='U'))
    RDET = {i: d for i, d in zip(RO.image_id, dets)}
RO['det_status'] = [RDET[i]['status'] for i in RO.image_id]
gal_r = RO.role.eq('gallery').to_numpy()
var_q = variants(RO[~gal_r], RDET, ['clean'])
var_g = variants(RO[gal_r], RDET)
ROF_E = {}
for key, var, sel in [('q', var_q, ~gal_r), ('g', var_g, gal_r)]:
    e = embed_all(var)
    for (m, k), v in e.items():
        full = ROF_E.setdefault((m, k), np.zeros((len(RO), 512), np.float32)); full[sel] = v
for m in ['B0', 'B1', 'ARC']:            # query chỉ có ảnh thật: biến thể "clean" của query là ảnh gốc
    for k in ['mask', 'sunglasses']:
        ROF_E[(m, k)][~gal_r] = ROF_E[(m, 'clean')][~gal_r]
FRAME_IDS = RO.canonical_identity.to_numpy()
Qr = RO[~gal_r].reset_index(drop=True); known_r = Qr.role.eq('known_query').to_numpy()
PRED = {}
for s in SYSTEMS:
    pred, score = system_scores(s, ROF_E, gal_r, [('clean', ~gal_r)], CALIB[s]['w'])
    q = Qr[['image_id', 'canonical_identity', 'role', 'condition', 'det_status']].copy()
    q['top_identity'], q['top_score'] = pred, score
    q['is_known'] = known_r; q['top1_correct'] = known_r & (pred == q.canonical_identity.to_numpy())
    q['accepted'] = score >= np.float32(CALIB[s]['threshold'])
    q['correct_accepted'] = q.accepted & q.top1_correct; q['unknown_accepted'] = q.accepted & ~q.is_known
    PRED[s] = q
save(pd.concat([q.assign(system=s) for s, q in PRED.items()]), '03_rof_predictions')
rows = []
for s, q in PRED.items():
    for cond in ['tất cả', 'clean', 'masked', 'sunglasses']:
        f = q if cond == 'tất cả' else q[q.condition.eq(cond)]
        k = f.is_known
        rows.append({'hệ thống': s, 'điều kiện': cond, 'query quen': int(k.sum()), 'DIR': float(f.correct_accepted[k].mean()),
                     'Rank-1': float(f.top1_correct[k].mean()), 'FPIR': float(f.unknown_accepted[~k].mean()) if (~k).any() else np.nan})
ROF_RES = save(pd.DataFrame(rows), '04_rof_results')
display(RO.groupby('condition').det_status.value_counts().unstack(fill_value=0))
display(ROF_RES.pivot(index='điều kiện', columns='hệ thống', values=['DIR', 'Rank-1', 'FPIR']).mul(100).round(1))
'''),
    ('code', '''
known_ids = sorted(Qr[known_r].canonical_identity.unique())


def boot(qa, qb, col, cond=None, reps=2000):
    def arr(q):
        f = q if cond is None else q[q.condition.eq(cond)]
        return f[f.is_known].groupby('canonical_identity').agg(h=(col, 'sum'), n=('image_id', 'size')).reindex(known_ids, fill_value=0).to_numpy(float)
    a, b = arr(qa), arr(qb); rng = np.random.default_rng(SEED); d = []
    for _ in range(reps):
        i = rng.integers(0, len(known_ids), len(known_ids))
        d.append(a[i, 0].sum() / max(a[i, 1].sum(), 1) - b[i, 0].sum() / max(b[i, 1].sum(), 1))
    return a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum(), *np.quantile(d, [.025, .975])


rows = []
for a, b in [('B1', 'B0'), ('B2', 'B1'), ('B3', 'B2'), ('B3', 'B0')]:
    for col, label in [('correct_accepted', 'DIR'), ('top1_correct', 'Rank-1')]:
        for cond in [None, 'masked', 'sunglasses']:
            p, lo, hi = boot(PRED[a], PRED[b], col, cond)
            rows.append({'so sánh': f'{a} − {b}', 'chỉ số': label, 'điều kiện': cond or 'tất cả', 'điểm': p, 'ci95 thấp': lo, 'ci95 cao': hi})
ROF_CI = save(pd.DataFrame(rows), '05_rof_ladder_CI')
display(ROF_CI.assign(**{c: lambda d, c=c: (100 * d[c]).round(1) for c in ['điểm', 'ci95 thấp', 'ci95 cao']}))
'''),
    ('md', '''
## 5. MFR2: xác minh 1:1 trên khẩu trang thật

848 cặp chính thức. B0/B1: cosine giữa hai ảnh. B2/B3: so thêm mỗi ảnh với bản đeo khẩu trang giả và kính giả của ảnh kia, lấy điểm lớn nhất (không cần biết trước ảnh nào đang đeo khẩu trang). Ngưỡng là ngưỡng đã khóa (chọn cho so khớp 1:N), nên accuracy tại ngưỡng chỉ để tham khảo; AUC/EER không cần ngưỡng.
'''),
    ('code', '''
EX = pd.read_csv(PROT / '07_external_image_roles_draft.csv', dtype=str, keep_default_na=False)
MF = EX[EX.source.eq('MFR2')].sort_values('image_id').reset_index(drop=True)
MF['rel'] = MF.path.str.replace('data/raw/mfr2/mfr2/', '', regex=False)
MF['occlusion_seed'] = [str(seed_of(SEED, 'test-occ', x)) for x in MF.image_id]
PAIRS = pd.read_csv(PROT / '08_mfr2_official_pairs_preserved.csv', dtype=str, keep_default_na=False)
y = PAIRS.same_person.eq('True').to_numpy()
assert all(sha(ROOT / r.path) == r.sha256 for r in MF.itertuples())
MDET = {r.image_id: aligner.detect(Image.open(ROOT / r.path).convert('RGB')) for r in MF.itertuples()}
MF_E = embed_all(variants(MF, MDET))
pos = {x: i for i, x in enumerate(MF.rel)}; ia = PAIRS.a.map(pos).to_numpy(); ib = PAIRS.b.map(pos).to_numpy()


def pair_score(s):
    fn = 'B0' if s == 'B0' else 'B1'
    def sim(ka, kb):
        f = (MF_E[(fn, ka)][ia] * MF_E[(fn, kb)][ib]).sum(1)
        if s != 'B3':
            return f
        w = CALIB['B3']['w']; return w * f + (1 - w) * (MF_E[('ARC', ka)][ia] * MF_E[('ARC', kb)][ib]).sum(1)
    if s in ('B0', 'B1'):
        return sim('clean', 'clean')
    return np.max(np.stack([sim('clean', 'clean')] + [sim('clean', k) for k in KINDS[1:]] + [sim(k, 'clean') for k in KINDS[1:]]), axis=0)


def auc_score(s, y):
    order = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def eer(s, y):
    ts = np.unique(s); far = np.array([(s[~y] >= t).mean() for t in ts]); frr = np.array([(s[y] < t).mean() for t in ts])
    i = np.argmin(np.abs(far - frr)); return float((far[i] + frr[i]) / 2)


SC = {s: pair_score(s) for s in SYSTEMS}
mrows = []
for s in SYSTEMS:
    acc = SC[s] >= CALIB[s]['threshold']
    mrows.append({'hệ thống': s, 'AUC': auc_score(SC[s], y), 'EER': eer(SC[s], y), 'accuracy@ngưỡng': float((acc == y).mean()),
                  'TAR@ngưỡng': float(acc[y].mean()), 'FAR@ngưỡng': float(acc[~y].mean())})
MF_RES = save(pd.DataFrame(mrows), '06_mfr2_results')
rng = np.random.default_rng(SEED); boots = {s: [] for s in SYSTEMS}
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    for s in SYSTEMS:
        boots[s].append(auc_score(SC[s][i], y[i]))
mci = []
for a, b in [('B1', 'B0'), ('B2', 'B1'), ('B3', 'B2'), ('B3', 'B0')]:
    d = np.array(boots[a]) - np.array(boots[b])
    mci.append({'so sánh': f'{a} − {b}', 'AUC điểm': auc_score(SC[a], y) - auc_score(SC[b], y), 'ci95 thấp': np.quantile(d, .025), 'ci95 cao': np.quantile(d, .975)})
MF_CI = save(pd.DataFrame(mci), '06_mfr2_AUC_CI')
display(MF_RES.round(4)); display(MF_CI.round(4))
'''),
    ('md', '''
## 6. Tổng hợp
'''),
    ('code', '''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={'width_ratios': [3, 1.1]})
conds = [('tất cả', 'ROF\\nmọi query'), ('clean', 'ROF\\nrõ mặt'), ('masked', 'ROF\\nkhẩu trang'), ('sunglasses', 'ROF\\nkính râm')]
w = .2
for i, s in enumerate(SYSTEMS):
    vals = [100 * ROF_RES[(ROF_RES['hệ thống'] == s) & (ROF_RES['điều kiện'] == c)].DIR.iloc[0] for c, _ in conds]
    bars = axes[0].bar(np.arange(len(conds)) + (i - 1.5) * (w + .01), vals, w, color=COLORS[s], label=s, zorder=3)
    axes[0].bar_label(bars, fmt='%.0f', fontsize=8, padding=2, color='#52514e')
fp = ROF_RES[ROF_RES['điều kiện'] == 'tất cả'].set_index('hệ thống').FPIR
axes[0].set_xticks(np.arange(len(conds)), [l for _, l in conds]); axes[0].set_ylim(0, 105)
axes[0].set(ylabel='DIR tại ngưỡng đã khóa (%)', title='ROF: tỷ lệ nhận đúng người quen (FPIR ' + ' / '.join(f'{s} {100 * fp[s]:.1f}%' for s in SYSTEMS) + ')')
axes[0].legend(frameon=False, ncols=4, loc='upper right')
mf = MF_RES.set_index('hệ thống')
bars = axes[1].bar(SYSTEMS, [100 * mf.loc[s, 'accuracy@ngưỡng'] for s in SYSTEMS], color=[COLORS[s] for s in SYSTEMS], zorder=3)
axes[1].bar_label(bars, fmt='%.0f', fontsize=8, padding=2, color='#52514e'); axes[1].set_ylim(0, 105)
axes[1].set(title='MFR2: độ chính xác tại ngưỡng khóa', ylabel='%')
axes[1].set_xlabel('AUC: ' + ' / '.join(f"{mf.loc[s, 'AUC']:.3f}" for s in SYSTEMS), fontsize=8, color='#52514e')
fig.tight_layout(); finish(fig, '01_ladder_B0_B3')
'''),
    ('code', '''
r = ROF_RES.set_index(['hệ thống', 'điều kiện']); m = MF_RES.set_index('hệ thống'); c = ROF_CI.set_index(['so sánh', 'chỉ số', 'điều kiện'])
fmt = lambda k: '{:+.1f} điểm [{:+.1f}; {:+.1f}]'.format(*(100 * c.loc[k, x] for x in ['điểm', 'ci95 thấp', 'ci95 cao'])).replace('.', ',')
lines = [f"| {s} | {pct(r.loc[(s, 'tất cả'), 'DIR'])} | {pct(r.loc[(s, 'masked'), 'DIR'])} | {pct(r.loc[(s, 'sunglasses'), 'DIR'])} | "
         f"{pct(r.loc[(s, 'tất cả'), 'FPIR'])} | {pct(r.loc[(s, 'tất cả'), 'Rank-1'])} | {m.loc[s, 'AUC']:.3f} |" for s in SYSTEMS]
json.dump({'checks': 'PASS', 'lock': LOCK_STATUS, 'calibration': CALIB, 'dev': DEV_TABLE.to_dict('records'), 'rof': ROF_RES.to_dict('records'),
           'rof_ci': ROF_CI.to_dict('records'), 'mfr2': MF_RES.to_dict('records'), 'mfr2_ci': MF_CI.to_dict('records'),
           'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - START},
          open(OUT / 'execution_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
display(Markdown(f"""## 7. Kết luận

| Hệ thống | ROF DIR mọi query | ROF DIR khẩu trang | ROF DIR kính râm | ROF FPIR | ROF Rank-1 | MFR2 AUC |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Hiệu từng bước trên ROF (DIR mọi query, CI 95% bootstrap theo người): B1 − B0 = {fmt(('B1 − B0', 'DIR', 'tất cả'))}; B2 − B1 = {fmt(('B2 − B1', 'DIR', 'tất cả'))}; B3 − B2 = {fmt(('B3 − B2', 'DIR', 'tất cả'))}; B3 − B0 = {fmt(('B3 − B0', 'DIR', 'tất cả'))}. Riêng khẩu trang: B3 − B0 = {fmt(('B3 − B0', 'DIR', 'masked'))}.

Ngưỡng và trọng số ghép (w = {CALIB['B3']['w']}) chọn trên dev LFW, khóa trong `{LOCK_PATH.name}` trước khi chấm. ROF có 178 query khẩu trang của 47 người nên CI riêng điều kiện này rộng. ArcFace (buffalo_l) là model công bố của InsightFace, chỉ dùng cho nghiên cứu phi thương mại.
"""))
'''),
    ('md', '''
## 8. Đọc kết quả

- **Trên khẩu trang thật (mục tiêu chính của đề tài), mỗi bước đều cải thiện có ý nghĩa.** DIR trên ROF khẩu trang: B0 27,0% → B1 39,3% → B2 53,9% → B3 83,1%. Cả ba bước đều có CI 95% nằm hẳn trên 0 (B1 − B0: +12,4 điểm [+3,7; +21,3]; B2 − B1: +14,6 [+4,6; +26,2]; B3 − B2: +29,2 [+20,5; +38,7]). Trên MFR2, độ chính xác tại ngưỡng khóa tăng 75,6% → 78,0% → 78,9% → 97,6%.
- **B1 và B2 có cái giá.** B1 kém B0 trên kính râm thật (−5,6 điểm DIR) và Rank-1 mọi query giảm 2,8 điểm: fine-tune trên 3.734 ảnh LFW vẫn làm lệch một chút khỏi VGGFace2. B2 đặt ngưỡng cao hơn để giữ FPIR trên dev có che, nên ảnh rõ mặt bị từ chối nhiều hơn (DIR rõ mặt 92,0% → 86,1%). Vì vậy tính trên mọi query, B1 và B2 chưa hơn B0 (74,9% → 73,6% → 70,9%).
- **B3 là bước mạnh nhất và không có cái giá đó:** hơn B0 ở mọi điều kiện (mọi query +19,2 điểm [+16,8; +21,7]) với FPIR thấp nhất (2,6%). Trọng số chọn trên dev là w = 0,25, tức ArcFace chiếm 75% điểm: phần lớn sức mạnh đến từ ArcFace, model được train trên WebFace600K (khoảng 600.000 người), lớn hơn nhiều so với dữ liệu nhóm có.
- **So với bài báo gốc:** Anwar & Raychowdhury (2020) train lại FaceNet từ đầu trên khoảng 700.000 ảnh có khẩu trang giả và báo TPR tăng khoảng 38%. Nhóm chỉ fine-tune nhẹ trên vài nghìn ảnh nên mức tăng của B1 nhỏ hơn, nhưng cùng chiều trên khẩu trang thật.
- **Giới hạn:** ROF có 178 query khẩu trang của 47 người nên CI rộng; khoảng 5% ảnh ROF MTCNN không thấy mặt và phải dùng nguyên ảnh (như nhau cho cả bốn hệ thống). Một seed; ngưỡng chọn trên LFW có che giả, chưa phải ảnh webcam.
'''),
], 'outputs_03_danh_gia')
