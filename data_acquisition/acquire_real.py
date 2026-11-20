"""Acquire ROF from author URL annotations and MFR2 from the author's Drive.
Never execute downloaded scripts. Originals are immutable; author-box crops are
benchmark reconstruction for EDA, not the proposed model preprocessing.
"""
import argparse, concurrent.futures as cf, hashlib, io, json, pickle, time
import urllib.request, urllib.error, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
HEADERS = {'User-Agent': 'AcademicFaceDatasetAudit/1.0'}

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray','scalar'): np.core.multiarray.scalar,
                   ('numpy.core.multiarray','_reconstruct'): np.core.multiarray._reconstruct,
                   ('numpy','ndarray'): np.ndarray, ('numpy','dtype'): np.dtype}
        if (module,name) in allowed: return allowed[(module,name)]
        raise pickle.UnpicklingError(f'Forbidden global: {module}.{name}')

def fetch(url, path, timeout=15, max_bytes=30_000_000):
    path = Path(path)
    if path.exists(): return {'status':'cached','bytes':path.stat().st_size}
    req=urllib.request.Request(url,headers=HEADERS)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        content_type=r.headers.get('Content-Type','')
        chunks=[];total=0
        while True:
            b=r.read(131072)
            if not b:break
            total+=len(b)
            if total>max_bytes:raise ValueError('download_size_limit')
            chunks.append(b)
        payload=b''.join(chunks)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.part');partial.write_bytes(payload);partial.replace(path)
    return {'status':'downloaded','bytes':total,'content_type':content_type}

def safe_extract(z, dest):
    dest=dest.resolve();dest.mkdir(parents=True,exist_ok=True)
    for name in z.namelist():
        candidate=(dest/name).resolve()
        if not candidate.is_relative_to(dest):raise ValueError('unsafe_zip_member')
    z.extractall(dest)

def rof(workers):
    src=DATA/'sources'/'rof';src.mkdir(parents=True,exist_ok=True)
    archive=src/'repository.zip'
    fetch('https://codeload.github.com/ekremerakin/RealWorldOccludedFaces/zip/refs/heads/main',archive,60)
    with zipfile.ZipFile(archive) as z:safe_extract(z,src)
    records=[]
    for p in sorted(src.rglob('*.pkl')):
        annotations=RestrictedUnpickler(io.BytesIO(p.read_bytes())).load()
        identity=p.stem.replace('_wearing_mask','').replace('_wearing_sunglasses','')
        for key,a in annotations.items():
            for j,face in enumerate(a.get('faces',[])):
                records.append({'source':'ROF','identity':identity,'condition':p.parent.name,
                    'annotation_file':p.relative_to(ROOT).as_posix(),'annotation_key':str(key),
                    'face_index':j,'url':a['url'],'author_box_xy_width_height':[int(x) for x in face['box']]})
    (src/'annotation_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    urls=sorted({r['url'] for r in records})
    print(f'ROF {len(records)} annotated faces, {len(urls)} unique URLs',flush=True)
    logpath=src/'download_log.jsonl'
    results={}
    def task(url):
        key=hashlib.sha256(url.encode()).hexdigest();p=DATA/'raw'/'rof_originals'/(key+'.img')
        result={'url':url,'original_path':p.relative_to(ROOT).as_posix(),'url_id':key}
        t=time.time()
        try:
            result.update(fetch(url,p,timeout=10,max_bytes=30_000_000))
            with Image.open(p) as im:
                im.load();result.update(width=im.width,height=im.height,format=im.format)
            result['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();result['ok']=True
        except Exception as e:result.update(ok=False,error=f'{type(e).__name__}: {str(e)[:240]}')
        result['seconds']=round(time.time()-t,3)
        return result
    with logpath.open('w',encoding='utf-8') as log,cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(task,urls),1):
            results[result['url']]=result;log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
            if i%100==0:print(f'ROF URLs {i}/{len(urls)} ok {sum(x["ok"] for x in results.values())}',flush=True)
    crop_rows=[]
    for i,r in enumerate(records):
        result=results[r['url']];row={**r,'download_ok':result['ok'],'original_path':result['original_path']}
        row['image_id']='ROF_'+hashlib.sha256((r['annotation_file']+'|'+r['annotation_key']+'|'+str(r['face_index'])).encode()).hexdigest()[:20]
        if result['ok']:
            try:
                with Image.open(ROOT/result['original_path']) as im:
                    im=im.convert('RGB');x,y,w,h=r['author_box_xy_width_height']
                    # Explicit clipping; unlike original script, include final row/column.
                    box=(max(0,x),max(0,y),min(im.width,x+w),min(im.height,y+h))
                    if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('invalid_author_box')
                    p=DATA/'benchmark_views'/'rof'/r['condition']/r['identity']/(row['image_id']+'.png')
                    p.parent.mkdir(parents=True,exist_ok=True);im.crop(box).save(p)
                    row.update(path=p.relative_to(ROOT).as_posix(),crop_ok=True,clipped_box=list(box),
                        box_clipped=list(box)!=[x,y,x+w,y+h])
            except Exception as e:row.update(crop_ok=False,error=str(e))
        else:row.update(crop_ok=False,error=result.get('error'))
        crop_rows.append(row)
    (src/'acquired_records.json').write_text(json.dumps(crop_rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ROF COMPLETE',len(crop_rows),'crops',sum(r['crop_ok'] for r in crop_rows),flush=True)

def mfr2():
    src=DATA/'sources'/'mfr2';src.mkdir(parents=True,exist_ok=True)
    for name in ['readme.md','datasets/download_links.txt','LICENSE']:
        fetch('https://raw.githubusercontent.com/aqeelanwar/MaskTheFace/master/'+name,src/Path(name).name)
    url='https://drive.usercontent.google.com/download?id=1ukk0n_srRqcsotK2MjlFPj7L0sXcR2fH&export=download&confirm=t'
    p=src/'mfr2.zip';info=fetch(url,p,60,500_000_000)
    if not zipfile.is_zipfile(p):raise ValueError('MFR2 response is not ZIP; inspect saved response')
    with zipfile.ZipFile(p) as z:safe_extract(z,DATA/'raw'/'mfr2')
    info.update(source_url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (src/'acquisition.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('MFR2 COMPLETE',info,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['rof','mfr2','both'],default='both');parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.source in ['mfr2','both']:mfr2()
    if args.source in ['rof','both']:rof(args.workers)
