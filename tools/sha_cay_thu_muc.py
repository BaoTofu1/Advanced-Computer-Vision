"""Ghi SHA256 mọi file trong một cây thư mục, không đi vào junction/symlink.

Dùng: python sha_cay_thu_muc.py <thư mục> <file .tsv đầu ra>
"""
import hashlib, os, sys
from pathlib import Path

root = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames
                         if not (Path(dirpath, d).is_symlink() or os.path.realpath(Path(dirpath, d)) != os.path.abspath(Path(dirpath, d))))
    for name in sorted(filenames):
        p = Path(dirpath, name)
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        rows.append(f'{p.relative_to(root).as_posix()}\t{p.stat().st_size}\t{h.hexdigest()}')
out.write_text('\n'.join(rows) + '\n', encoding='utf-8')
print(len(rows), 'files ->', out)
