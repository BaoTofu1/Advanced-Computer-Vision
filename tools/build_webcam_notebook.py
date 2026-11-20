"""Dựng notebook bước 8B: chọn ngưỡng và test trên ảnh webcam của nhóm.

Các ô sau ô kiểm dữ liệu được bọc trong `if DATA_READY:` để notebook chạy trọn vẹn (và nói rõ
còn thiếu gì) khi nhóm chưa thu xong ảnh.
"""
from pathlib import Path
import textwrap

import nbformat as nbf

BASE = Path(__file__).resolve().parents[1]   # thư mục gốc của repo
ROOT = BASE
TARGET = ROOT / 'ban_giao_nhom/webcam/Hieu_chinh_va_test_webcam.ipynb'
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))


def gcode(s):
    body = textwrap.indent(textwrap.dedent(s).strip(), '    ')
    cells.append(nbf.v4.new_code_cell("if not DATA_READY:\n    display(Markdown(SKIP))\nelse:\n" + body))


md('''
# Bước 8B — Chọn ngưỡng và test trên ảnh webcam

**Đồ án CV nâng cao · 16/11/2026**

Bước 7 cho thấy ngưỡng chọn trên DigiFace không dùng được cho ảnh thật. Notebook này làm lại hai việc đó trên ảnh webcam của chính nhóm:

1. **Phiên 2 (hiệu chỉnh):** chọn ngưỡng cho từng model sao cho người lạ bị nhận nhầm không quá 5%.
2. **Phiên 3 (test):** khóa ngưỡng rồi mới chấm, theo từng điều kiện (rõ mặt, khẩu trang, kính, tay che, nghiêng, tối).

Gallery lấy từ phiên 1 (đăng ký). Ảnh được xử lý bằng đúng code của app demo (`demo/nhan_dien.py`): MTCNN phải thấy đúng một mặt; ảnh không thấy mặt vẫn nằm trong mẫu số và tính là không nhận ra.

Dữ liệu thu theo `HUONG_DAN_THU_ANH_WEBCAM.md`, bằng `demo/thu_anh_webcam.py`. Khi chưa đủ dữ liệu, mục 1 liệt kê còn thiếu gì và các mục sau tự bỏ qua.
''')

md('''
## 1. Kiểm dữ liệu đã thu

Cấu trúc cần có trong `data/raw/webcam/`:

```text
nguoi_tham_gia.csv      ma_nguoi, vai_tro (known/unknown), phien_tham_gia, dong_y (yes/no), ngay_xoa_du_kien
s1_dang_ky/<mã người>/<điều kiện>_<số>.jpg      chỉ người known
s2_hieu_chinh/<mã người>/...                    known + người lạ nhóm 1
s3_test/<mã người>/...                          known + người lạ nhóm 2 (khác nhóm 1)
```

Các kiểm "bắt buộc" phải đạt thì notebook mới chạy tiếp; "cảnh báo" không chặn nhưng làm kết quả kém tin cậy.
''')

code('''
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, time, platform, importlib.metadata, warnings
warnings.filterwarnings('ignore', message='IProgress not found')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import torch


def find_root():
    options = [Path(os.environ['CV_PROJECT_ROOT'])] if os.environ.get('CV_PROJECT_ROOT') else []
    options += [Path.cwd(), *Path.cwd().parents]
    for p in options:
        if (p / 'ban_giao_nhom/webcam/Hieu_chinh_va_test_webcam.ipynb').is_file():
            return p.resolve()
    raise FileNotFoundError('Mở notebook trong thư mục dự án hoặc đặt CV_PROJECT_ROOT')


ROOT = find_root()
HERE = ROOT / 'ban_giao_nhom/webcam'
REHEARSAL = os.environ.get('CV_WEBCAM_REHEARSAL') == '1'       # diễn tập code bằng ảnh không phải webcam
WEBCAM = Path(os.environ['CV_WEBCAM_DIR']) if os.environ.get('CV_WEBCAM_DIR') else ROOT / 'data/raw/webcam'
OUT = HERE / ('notebook_outputs_dien_tap' if REHEARSAL else 'notebook_outputs')
TAB, FIG = OUT / 'tables', OUT / 'figures'
LOCK_PATH = (OUT if REHEARSAL else HERE) / 'khoa_nguong_webcam.json'
DEMO_THRESHOLDS = ROOT / 'ban_giao_nhom/demo/nguong_webcam.json'
sys.path.insert(0, str(ROOT / 'ban_giao_nhom/demo'))
sys.path.insert(0, str(ROOT / 'ban_giao_nhom/cai_tien_B0_B3'))
from nhan_dien import open_rgb, IMAGE_SUFFIXES, MTCNN_CONFIG
from cai_tien import LadderRecognizer

SEED = 20261028
TARGET_FPIR = 0.05
MODELS = LadderRecognizer.SYSTEMS          # chuỗi cải tiến B0 → B3 (cai_tien_B0_B3/)
SESSIONS = {'s1_dang_ky': 'đăng ký', 's2_hieu_chinh': 'hiệu chỉnh', 's3_test': 'test'}
CONDITIONS = {'ro': 'Rõ mặt', 'khau_trang': 'Khẩu trang', 'kinh': 'Kính', 'tay_che': 'Tay che', 'nghieng': 'Nghiêng', 'toi': 'Tối'}
COLORS = {'B0': '#8a8984', 'B1': '#2a78d6', 'B2': '#1baf7a', 'B3': '#eb6834'}
SKIP = 'Bỏ qua: chưa có dữ liệu webcam hợp lệ (xem mục 1).'
START = time.perf_counter()
plt.rcParams.update({'font.family': 'Arial', 'figure.dpi': 110, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': .6})


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def save(d, name):
    TAB.mkdir(parents=True, exist_ok=True)
    d.to_csv(TAB / (name + '.csv'), index=False, encoding='utf-8-sig', float_format='%.17g'); return d


def vn(x, d=1):
    return f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def pct(x):
    return 'không có' if x is None or (isinstance(x, float) and np.isnan(x)) else vn(100 * x, 1) + '%'


def finish(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(FIG / (name + '.' + ext), bbox_inches='tight', dpi=160)
    display(fig); plt.close(fig)


# ---- Kiểm kê ----
checks = []


def check(name, ok, detail='', required=True):
    checks.append({'kiểm': name, 'đạt': bool(ok), 'mức': 'bắt buộc' if required else 'cảnh báo', 'chi tiết': detail})
    return bool(ok)


rows = []
for session in SESSIONS:
    for path in sorted((WEBCAM / session).glob('*/*')) if (WEBCAM / session).is_dir() else []:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            stem = path.stem
            rows.append({'image_id': f'{session}/{path.parent.name}/{path.name}', 'session': session, 'person': path.parent.name,
                         'condition': stem.rsplit('_', 1)[0] if '_' in stem else stem, 'path': path})
M = pd.DataFrame(rows, columns=['image_id', 'session', 'person', 'condition', 'path'])
people_path = WEBCAM / 'nguoi_tham_gia.csv'
PEOPLE = pd.read_csv(people_path, dtype=str, keep_default_na=False, encoding='utf-8-sig') if people_path.exists() else pd.DataFrame()
need_cols = ['ma_nguoi', 'vai_tro', 'dong_y']

check('Có thư mục dữ liệu webcam', WEBCAM.is_dir(), str(WEBCAM))
ok_people = check('Có nguoi_tham_gia.csv đủ cột', all(c in PEOPLE.columns for c in need_cols), ', '.join(need_cols))
for session, label in SESSIONS.items():
    check(f'Có ảnh phiên {label} ({session})', (M.session == session).any(), f'{int((M.session == session).sum())} ảnh')
if ok_people and len(M):
    role = dict(zip(PEOPLE.ma_nguoi, PEOPLE.vai_tro.str.strip().str.lower()))
    consent = dict(zip(PEOPLE.ma_nguoi, PEOPLE.dong_y.str.strip().str.lower()))
    M['role'] = M.person.map(role)
    missing = sorted(set(M.person) - set(role))
    check('Mọi người có ảnh đều có trong nguoi_tham_gia.csv', not missing, ', '.join(missing))
    check('Vai trò chỉ là known hoặc unknown', set(role.values()) <= {'known', 'unknown'}, ', '.join(sorted(set(role.values()))))
    no_consent = sorted(p for p in set(M.person) if consent.get(p) != 'yes')
    check('Mọi người có ảnh đều đã đồng ý (dong_y = yes)', not no_consent, ', '.join(no_consent))
    s1, s2, s3 = (M[M.session == s] for s in SESSIONS)
    bad = sorted(set(s1[s1.role == 'unknown'].person))
    check('Người lạ không có ảnh trong phiên đăng ký', not bad, ', '.join(bad))
    known_gallery = set(s1[s1.role == 'known'].person)
    not_enrolled = sorted(set(M[(M.role == 'known') & (M.session != 's1_dang_ky')].person) - known_gallery)
    check('Người known có ảnh ở phiên 2/3 đều có ảnh đăng ký', not not_enrolled, ', '.join(not_enrolled))
    overlap = sorted(set(s2[s2.role == 'unknown'].person) & set(s3[s3.role == 'unknown'].person))
    check('Người lạ của phiên 2 và phiên 3 là hai nhóm khác nhau', not overlap, ', '.join(overlap))
    check('Phiên 2 có ảnh người known và người lạ', (s2.role == 'known').any() and (s2.role == 'unknown').any(),
          f"{int((s2.role == 'known').sum())} known, {int((s2.role == 'unknown').sum())} lạ")
    check('Phiên 3 có ảnh người known', (s3.role == 'known').any(), f"{int((s3.role == 'known').sum())} ảnh")
    odd = sorted(set(M.condition) - set(CONDITIONS))
    check('Tên điều kiện đúng quy ước', not odd, ', '.join(odd))
    M['sha256'] = [sha(p) for p in M.path]
    dup = M[M.duplicated('sha256', keep=False)].image_id.tolist()
    check('Không có ảnh trùng nội dung giữa các thư mục', not dup, ', '.join(dup[:6]))
    few = sorted(p for p, n in s1.groupby('person').size().items() if n < 3)
    check('Mỗi người đăng ký có ít nhất 3 ảnh', not few, ', '.join(few), required=False)
    check('Có ít nhất 10 người known', len(known_gallery) >= 10, f'{len(known_gallery)} người', required=False)
    n_unknown_s2 = int((s2.role == 'unknown').sum())
    check('Phiên 2 có ít nhất 40 ảnh người lạ (FPIR 5% = 2 ảnh)', n_unknown_s2 >= 40, f'{n_unknown_s2} ảnh', required=False)
    check('Phiên 3 có ảnh người lạ để đo FPIR', (s3.role == 'unknown').any(), f"{int((s3.role == 'unknown').sum())} ảnh", required=False)
CHECKS = pd.DataFrame(checks)
DATA_READY = bool(CHECKS[CHECKS['mức'] == 'bắt buộc']['đạt'].all())
display(CHECKS)
if REHEARSAL:
    display(Markdown('**ĐANG DIỄN TẬP:** dữ liệu không phải ảnh webcam của nhóm. Kết quả chỉ để kiểm code, không dùng trong báo cáo.'))
if DATA_READY:
    inventory = (M.assign(phiên=M.session.map(SESSIONS)).groupby(['phiên', 'role', 'condition'])
                 .agg(ảnh=('image_id', 'size'), người=('person', 'nunique')).reset_index())
    save(inventory, '01_inventory'); display(inventory)
    display(Markdown(f'**Dữ liệu đủ điều kiện:** {len(M)} ảnh, {M[M.role == "known"].person.nunique()} người known, '
                     f'{M[M.role == "unknown"].person.nunique()} người lạ.'))
else:
    failed = CHECKS[(CHECKS['mức'] == 'bắt buộc') & ~CHECKS['đạt']]['kiểm'].tolist()
    display(Markdown('**Chưa chạy được.** Các kiểm bắt buộc chưa đạt: ' + '; '.join(failed) + '. Thu ảnh bằng `demo/thu_anh_webcam.py` '
                     'theo `HUONG_DAN_THU_ANH_WEBCAM.md`, điền `nguoi_tham_gia.csv`, rồi Restart Kernel → Run All.'))
''')

md('''
## 2. Tìm mặt và trích embedding

Mỗi ảnh đi qua MTCNN với cấu hình của bước 4/7, giống app demo. Chỉ ảnh có đúng một mặt mới được nhận diện; ảnh không thấy mặt hoặc thấy nhiều mặt tính là nhận sai. Bảng dưới cho biết detector hoạt động thế nào ở từng điều kiện; đây là một phần kết quả, không phải bước lọc dữ liệu.
''')

gcode('''
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
rec = LadderRecognizer('B0', device=DEVICE, root=ROOT)
IMAGES = [open_rgb(path) for path in M.path]
DETS = [rec.aligner.detect(im) for im in IMAGES]
M['status'] = [{'one_face': 'single_face', 'center_face': 'multiple_faces', 'fallback': 'no_face'}[d['status']] for d in DETS]
M['has_crop'] = M.status.eq('single_face')
ITEMS = {i: rec.prepare(IMAGES[i], DETS[i]) for i in np.flatnonzero(M.has_crop.to_numpy())}
detector = (M.assign(phiên=M.session.map(SESSIONS)).groupby(['phiên', 'condition'])
            .agg(ảnh=('image_id', 'size'), đúng_1_mặt=('has_crop', 'sum')).reset_index())
detector['tỷ lệ thấy đúng 1 mặt'] = (100 * detector['đúng_1_mặt'] / detector['ảnh']).round(1)
save(detector, '02_detector_by_condition'); display(detector)
display(M.status.value_counts().rename('số ảnh').to_frame())
''')

md('''
## 3. Gallery từ phiên đăng ký

Đăng ký giống app demo (`LadderRecognizer.enroll`): template mỗi người = trung bình embedding các ảnh đăng ký, chuẩn hóa lại; B2/B3 thêm template từ chính các ảnh đó đeo khẩu trang và kính giả. Người không còn ảnh đăng ký hợp lệ thì không đăng ký được; ảnh của họ ở phiên 2/3 vẫn nằm trong mẫu số.
''')

gcode('''
g = M[(M.session == 's1_dang_ky') & (M.role == 'known')]
enroll = g.groupby('person').agg(ảnh_đăng_ký=('image_id', 'size'), ảnh_dùng_được=('has_crop', 'sum')).reset_index()
people = {p: [IMAGES[i] for i in f.index if M.has_crop[i]] for p, f in g.groupby('person')}
rec.enroll({p: ims for p, ims in people.items() if ims})
GALLERY_NAMES = rec.names
save(enroll, '03_enrollment'); display(enroll)
display(Markdown(f'Đăng ký được **{len(GALLERY_NAMES)}/{len(enroll)} người**.'))


def score(session, m):
    rec.set_system(m)
    f = M[M.session == session]
    has = f.has_crop.to_numpy(); idx = f.index.to_numpy()[has]
    top = np.zeros(len(f), int); best = np.full(len(f), -np.inf, np.float32)
    if len(idx):
        fn, arc = rec.embed([ITEMS[i] for i in idx])
        sim = rec.scores(fn, arc)
        top[has] = sim.argmax(axis=1); best[has] = sim[np.arange(len(idx)), sim.argmax(axis=1)]
    q = f[['image_id', 'session', 'person', 'role', 'condition', 'status']].copy()
    q['is_known'] = q.role.eq('known')
    q['top_identity'] = np.where(has, np.asarray(GALLERY_NAMES)[top], '')
    q['top_score'] = best.astype(np.float32)
    q['top1_correct'] = q.is_known & has & (q.top_identity == q.person)
    return q.reset_index(drop=True)


def apply_threshold(q, threshold):
    q = q.copy()
    q['accepted'] = q.top_score.to_numpy(np.float32) >= np.float32(threshold)
    q['correct_accepted'] = q.accepted & q.top1_correct
    q['unknown_accepted'] = q.accepted & ~q.is_known
    return q


def rates(q):
    k, u = q.is_known, ~q.is_known
    return {'ảnh known': int(k.sum()), 'ảnh lạ': int(u.sum()),
            'DIR': float(q.correct_accepted[k].mean()) if k.any() else np.nan,
            'Rank-1': float(q.top1_correct[k].mean()) if k.any() else np.nan,
            'FPIR': float(q.unknown_accepted[u].mean()) if u.any() else np.nan,
            'thấy đúng 1 mặt': float(q.status.eq('single_face').mean())}


def choose_threshold(q):
    """DIR lớn nhất trong các ngưỡng có FPIR <= mục tiêu; hòa thì lấy ngưỡng cao hơn."""
    known = q.is_known.to_numpy(); correct = q.top1_correct.to_numpy(); s = q.top_score.to_numpy(np.float32)
    cand = np.r_[np.unique(s[np.isfinite(s)]).astype(np.float64), np.inf]
    hit = np.sort(s[known & correct]); unk = np.sort(s[~known])
    DIR = (len(hit) - np.searchsorted(hit, cand, side='left')) / max(known.sum(), 1)
    FPIR = (len(unk) - np.searchsorted(unk, cand, side='left')) / max((~known).sum(), 1)
    ok = FPIR <= TARGET_FPIR
    return float(cand[ok & (DIR == DIR[ok].max())].max())


def person_bootstrap(qs, col='correct_accepted', reps=2000):
    """CI 95% theo người cho tỷ lệ `col` trên ảnh known của từng model và hiệu giữa các model."""
    persons = sorted(set().union(*[set(q[q.is_known].person) for q in qs.values()]))
    if len(persons) < 5:
        return pd.DataFrame()
    arr = {m: q[q.is_known].groupby('person').agg(h=(col, 'sum'), n=('image_id', 'size')).reindex(persons, fill_value=0).to_numpy(float)
           for m, q in qs.items()}
    rng = np.random.default_rng(SEED); draws = {m: [] for m in qs}
    for _ in range(reps):
        i = rng.integers(0, len(persons), len(persons))
        for m, a in arr.items():
            draws[m].append(a[i, 0].sum() / max(a[i, 1].sum(), 1))
    draws = {m: np.array(v) for m, v in draws.items()}
    out = [{'so sánh': m, 'điểm': arr[m][:, 0].sum() / arr[m][:, 1].sum(),
            'ci95 thấp': np.quantile(draws[m], .025), 'ci95 cao': np.quantile(draws[m], .975)} for m in qs]
    for a, b in [('B1', 'B0'), ('B2', 'B1'), ('B3', 'B2'), ('B3', 'B0')]:
        d = draws[a] - draws[b]
        out.append({'so sánh': f'{a}-{b}', 'điểm': out[MODELS.index(a)]['điểm'] - out[MODELS.index(b)]['điểm'],
                    'ci95 thấp': np.quantile(d, .025), 'ci95 cao': np.quantile(d, .975)})
    return pd.DataFrame(out).assign(số_người=len(persons))
''')

md('''
## 4. Chọn ngưỡng trên phiên hiệu chỉnh

Với mỗi model, quét mọi ngưỡng trên ảnh phiên 2 và lấy ngưỡng cho DIR cao nhất trong các ngưỡng có FPIR ≤ 5%. Ảnh người lạ ít thì FPIR nhảy từng bước lớn (40 ảnh: mỗi ảnh là 2,5%), nên con số 5% chỉ là mục tiêu trên mẫu này, không phải bảo đảm.
''')

gcode('''
CAL = {m: score('s2_hieu_chinh', m) for m in MODELS}
THRESHOLDS = {m: choose_threshold(CAL[m]) for m in MODELS}
cal_rows = [{'model': m, 'ngưỡng': THRESHOLDS[m], **rates(apply_threshold(CAL[m], THRESHOLDS[m])),
             'người known': CAL[m][CAL[m].is_known].person.nunique(), 'người lạ': CAL[m][~CAL[m].is_known].person.nunique()} for m in MODELS]
CAL_TABLE = save(pd.DataFrame(cal_rows), '04_calibration_session2')
show = CAL_TABLE.copy()
for c in ['DIR', 'Rank-1', 'FPIR', 'thấy đúng 1 mặt']:
    show[c] = (100 * show[c]).round(1)
display(show)
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
bins = np.linspace(-.2, 1, 37)
for ax, m in zip(axes, MODELS):
    q = CAL[m][np.isfinite(CAL[m].top_score)]
    ax.hist(q[q.top1_correct].top_score, bins=bins, color='#2a78d6', alpha=.8, label='Người quen, đúng tên', zorder=3)
    ax.hist(q[~q.is_known].top_score, bins=bins, color='#eb6834', alpha=.8, label='Người lạ', zorder=3)
    ax.axvline(THRESHOLDS[m], color='#0b0b0b', lw=1.5, ls='--')
    ax.text(THRESHOLDS[m], ax.get_ylim()[1] * .97, f' ngưỡng {THRESHOLDS[m]:.3f}', va='top', fontsize=9)
    ax.set(title=m, xlabel='Cosine cao nhất với gallery')
axes[0].set_ylabel('Số ảnh phiên 2'); axes[0].legend(frameon=False, fontsize=8, loc='upper left')
fig.suptitle('Phân bố điểm trên phiên hiệu chỉnh và ngưỡng được chọn', y=1.03); fig.tight_layout(); finish(fig, '01_calibration_scores')
''')

md('''
## 5. Khóa ngưỡng

Ngưỡng, SHA danh sách ảnh và SHA model được ghi vào `khoa_nguong_webcam.json` trước khi chấm phiên 3. Nếu file đã có, notebook kiểm lại cho khớp. Thu thêm ảnh rồi muốn hiệu chỉnh lại thì phải tự xóa file khóa; khi đó kết quả phiên 3 đã xem trước đó không còn là test độc lập.

Ngưỡng cũng được chép sang `demo/nguong_webcam.json` để app demo tự dùng.
''')

gcode('''
lock_b = json.loads((ROOT / 'ban_giao_nhom/cai_tien_B0_B3/khoa_cau_hinh_B0_B3.json').read_text(encoding='utf-8'))
LOCK = {'version': 'webcam_threshold_v1', 'target_FPIR': TARGET_FPIR,
        'thresholds': {m: THRESHOLDS[m] for m in MODELS},
        'calibration_manifest_sha256': digest(M[M.session != 's3_test'][['image_id', 'sha256']].to_csv(index=False)),
        'models': lock_b['models'], 'w_facenet_B3': lock_b['systems']['B3']['w_facenet'],
        'mtcnn': MTCNN_CONFIG, 'gallery_persons': GALLERY_NAMES,
        'rule': 'max DIR trên phiên 2 với FPIR <= mục tiêu; hòa lấy ngưỡng cao hơn; ảnh không có đúng 1 mặt tính là không nhận'}
if LOCK_PATH.exists():
    old = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    same = all(old[k] == json.loads(json.dumps(LOCK))[k] for k in LOCK)
    assert same, ('Dữ liệu phiên 1/2 hoặc cấu hình đã đổi so với file khóa. Nếu chủ động hiệu chỉnh lại, xóa '
                  f'{LOCK_PATH.name} rồi chạy lại, và ghi rõ trong báo cáo.')
    lock_status = 'file khóa có sẵn, khớp'
else:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps({**LOCK, 'locked_at_utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2), encoding='utf-8')
    lock_status = 'vừa ghi file khóa'
if not REHEARSAL:
    DEMO_THRESHOLDS.write_text(json.dumps({'thresholds': LOCK['thresholds'], 'source': 'webcam phiên 2',
                                           'lock_sha256': sha(LOCK_PATH)}, ensure_ascii=False, indent=2), encoding='utf-8')
display(Markdown(f'**{lock_status}:** `{LOCK_PATH.name}`. Ngưỡng: ' + ', '.join(f'{m} = {THRESHOLDS[m]:.4f}' for m in MODELS)))
''')

md('''
## 6. Test trên phiên 3

Ngưỡng giữ nguyên từ mục 5. DIR = ảnh người quen được nhận đúng tên và vượt ngưỡng / mọi ảnh người quen (kể cả ảnh detector không thấy mặt). FPIR = ảnh người lạ bị gán một tên / mọi ảnh người lạ. Khoảng tin cậy lấy bằng bootstrap theo người khi có từ 5 người known trở lên.
''')

gcode('''
TEST = {m: apply_threshold(score('s3_test', m), THRESHOLDS[m]) for m in MODELS}
rows = []
for m in MODELS:
    q = TEST[m]
    overall = rates(q)
    rows.append({'model': m, 'điều kiện': 'Tất cả', **overall})
    for cond in [c for c in CONDITIONS if (q.condition == c).any()]:
        rows.append({'model': m, 'điều kiện': CONDITIONS[cond], **rates(q[q.condition == cond])})
RESULT = save(pd.DataFrame(rows), '05_test_session3')
PRED = pd.concat([TEST[m].assign(model=m, threshold=THRESHOLDS[m]) for m in MODELS], ignore_index=True)
save(PRED[['model', 'image_id', 'person', 'role', 'condition', 'status', 'top_identity', 'top_score', 'threshold',
           'accepted', 'top1_correct', 'correct_accepted', 'unknown_accepted']], '05_predictions_session3')
CI = person_bootstrap(TEST)
if len(CI):
    save(CI, '05_test_DIR_CI')
show = RESULT.copy()
for c in ['DIR', 'Rank-1', 'FPIR', 'thấy đúng 1 mặt']:
    show[c] = (100 * show[c]).round(1)
display(show)
if len(CI):
    display(CI.assign(**{c: lambda d, c=c: (100 * d[c]).round(1) for c in ['điểm', 'ci95 thấp', 'ci95 cao']}))
else:
    display(Markdown('Chưa đủ 5 người known ở phiên 3 để tính khoảng tin cậy theo người.'))

conds = [c for c in ['Tất cả'] + list(CONDITIONS.values()) if (RESULT['điều kiện'] == c).any()]
fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(conds) + 2), 4.2)); w = .2
for i, m in enumerate(MODELS):
    vals = [100 * RESULT[(RESULT.model == m) & (RESULT['điều kiện'] == c)].DIR.iloc[0] for c in conds]
    bars = ax.bar(np.arange(len(conds)) + (i - 1.5) * (w + .01), vals, w, color=COLORS[m], label=m, zorder=3)
    ax.bar_label(bars, fmt='%.0f', fontsize=8, padding=2, color='#52514e')
ax.set_xticks(np.arange(len(conds)), conds); ax.set_ylim(0, 108)
ax.set(ylabel='DIR (%)', title='Webcam phiên 3: tỷ lệ nhận đúng người quen tại ngưỡng đã khóa')
ax.legend(frameon=False, ncols=3, loc='upper right'); fig.tight_layout(); finish(fig, '02_test_DIR_by_condition')
''')

gcode('''
overall = RESULT[RESULT['điều kiện'] == 'Tất cả'].set_index('model')
best = overall.DIR.idxmax()
lines = ['## 7. Kết luận', '']
if REHEARSAL:
    lines += ['**DIỄN TẬP, không phải kết quả webcam.**', '']
lines.append(f"- **Dữ liệu:** {M[M.role == 'known'].person.nunique()} người known, {len(GALLERY_NAMES)} người đăng ký được; "
             f"phiên 3 có {int(overall.loc['B0', 'ảnh known'])} ảnh người quen và {int(overall.loc['B0', 'ảnh lạ'])} ảnh người lạ.")
for m in MODELS:
    r = overall.loc[m]
    lines.append(f"- **{m}** (ngưỡng {THRESHOLDS[m]:.3f}): DIR {pct(r.DIR)}, Rank-1 {pct(r['Rank-1'])}, FPIR {pct(r.FPIR)}.")
lines.append(f"- **Detector:** thấy đúng một mặt ở {pct(float(TEST['B0'].status.eq('single_face').mean()))} ảnh phiên 3; theo điều kiện xem bảng mục 2.")
if len(CI):
    d = CI.set_index('so sánh')
    lines.append(f"- **So model (DIR, bootstrap theo {int(d['số_người'].iloc[0])} người):** "
                 + '; '.join(f"{k} = {100 * d.loc[k, 'điểm']:+.1f} điểm [{100 * d.loc[k, 'ci95 thấp']:+.1f}; {100 * d.loc[k, 'ci95 cao']:+.1f}]"
                             for k in ['B1-B0', 'B2-B1', 'B3-B2', 'B3-B0']).replace('.', ','))
lines.append(f"- **Model có DIR cao nhất trên webcam: {best}.** App demo đọc ngưỡng từ `demo/nguong_webcam.json`.")
lines += ['', 'Số người ít nên khoảng tin cậy rộng; kết quả đúng cho camera, ánh sáng và nhóm người đã thu, chưa phải cho mọi điều kiện. '
          'Hệ thống không có chống giả mạo.']
summary = {'checks': 'PASS', 'rehearsal': REHEARSAL, 'webcam_dir': str(WEBCAM), 'images': len(M), 'thresholds': THRESHOLDS,
           'lock_file': LOCK_PATH.name, 'lock_status': lock_status, 'calibration': CAL_TABLE.to_dict('records'),
           'test': RESULT.to_dict('records'), 'gallery_persons': len(GALLERY_NAMES),
           'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': DEVICE,
                           'facenet_pytorch': importlib.metadata.version('facenet-pytorch')},
           'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - START}
(OUT / 'execution_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
display(Markdown(chr(10).join(lines)))
''')

md('''
## 8. Đầu ra

- `khoa_nguong_webcam.json`: ngưỡng đã khóa trước khi chấm phiên 3; `../demo/nguong_webcam.json`: bản cho app demo.
- `notebook_outputs/tables/`: `01_inventory`, `02_detector_by_condition`, `03_enrollment`, `04_calibration_session2`, `05_test_session3`, `05_predictions_session3`, `05_test_DIR_CI`.
- `notebook_outputs/figures/`: phân bố điểm phiên 2 và DIR theo điều kiện phiên 3.

Ảnh webcam và `nguoi_tham_gia.csv` là dữ liệu cá nhân: chỉ nhóm giữ, không đưa vào ZIP, không đăng lên mạng, xóa theo ngày đã hứa.
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3.11 (.venv đồ án)', 'language': 'python', 'name': 'cv-claude-full'},
    'language_info': {'name': 'python', 'version': '3.11.9'}})
TARGET.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, TARGET)
print(TARGET, {'cells': len(cells), 'code_cells': sum(c.cell_type == 'code' for c in cells)})
