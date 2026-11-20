"""Restart + Run All một notebook trong bản sao bằng kernel .venv của đồ án, lưu output thật.

Dùng (từ thư mục gốc repo): python tools/run_notebook.py <notebook> [--out <file .ipynb khác>]
Biến môi trường CV_TRAIN_MODE / CV_FULL_SMOKE được chuyển nguyên sang kernel.
"""
from pathlib import Path
import argparse, asyncio, json, os, sys, time, traceback
import nbformat
from nbclient import NotebookClient

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent                     # thư mục gốc của repo
# Python của môi trường dự án: biến CV_PYTHON nếu có, nếu không dùng chính Python đang chạy script này.
VENV_PY = Path(os.environ.get('CV_PYTHON') or sys.executable)
ap = argparse.ArgumentParser()
ap.add_argument('notebook'); ap.add_argument('--out')
args = ap.parse_args()
target = Path(args.notebook).resolve()
assert ROOT in target.parents, 'Chỉ chạy notebook trong bản sao'
out = Path(args.out).resolve() if args.out else target

spec = TOOLS / 'jupyter/kernels/cv-claude-full'
spec.mkdir(parents=True, exist_ok=True)
(spec / 'kernel.json').write_text(json.dumps({
    'argv': [str(VENV_PY), '-X', 'utf8', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python 3.11 (.venv đồ án)', 'language': 'python',
    'env': {'PYTHONUTF8': '1', 'CV_PROJECT_ROOT': str(ROOT), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(TOOLS / 'jupyter')
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

nb = nbformat.read(target, as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []; cell.execution_count = None
code_cells = sum(c.cell_type == 'code' for c in nb.cells)
started = time.time(); errors = []; executed = 0


def after(cell, cell_index, **kw):
    global executed
    if cell.cell_type == 'code':
        executed += 1
        print(f'DONE cell {cell_index} [{cell.execution_count}] {time.time() - started:.1f}s', flush=True)


try:
    NotebookClient(nb, kernel_name='cv-claude-full', timeout=None,
                   resources={'metadata': {'path': str(target.parent)}}, on_cell_executed=after).execute()
except Exception as exc:
    errors.append(f'{type(exc).__name__}: {str(exc)[-3000:]}')
    traceback.print_exc()
finally:
    nbformat.write(nb, out)
    report = {'notebook': target.name, 'output': str(out), 'fresh_kernel': True, 'code_cells': code_cells,
              'executed_cells': executed, 'errors': errors,
              'env': {k: os.environ.get(k) for k in ['CV_TRAIN_MODE', 'CV_FULL_SMOKE']},
              'elapsed_seconds': round(time.time() - started, 1)}
    (out.parent / f'{out.stem}_execution_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
if errors:
    raise SystemExit(1)
