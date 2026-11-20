"""One conservative retry pass for failed ROF requests; preserve original audit logs.

Only timeout/disconnect/network-timeout and HTTP 5xx are eligible. HTTP 525/526
represent TLS failures and are excluded. No 429/401/403/404, DNS or TLS bypass.
Uses acquire_real.fetch unchanged, 4 workers, timeout 15 s, at most 80 URLs.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from PIL import Image
from acquire_real import ROOT, DATA, fetch

SOURCE = DATA / "sources/rof"


def eligible(row):
    if row.get("ok") or row.get("retried"):
        return False
    error = row.get("error", "")
    if error.startswith(("TimeoutError:", "RemoteDisconnected:")):
        return True
    if error.startswith("URLError:") and "timed out" in error.lower():
        return not any(s in error.lower() for s in ("ssl", "tls", "cert", "getaddrinfo"))
    match = re.match(r"HTTPError: HTTP Error (5\d\d):", error)
    return bool(match and int(match.group(1)) not in (525, 526)
                and not any(s in error.lower() for s in ("ssl", "tls", "certificate")))


def attempt(old):
    row = dict(old)
    row.update(retried=True, attempts=2, initial_error=old.get("error"),
               retry_at_utc=datetime.now(timezone.utc).isoformat())
    t = time.monotonic()
    try:
        info = fetch(old["url"], ROOT / old["original_path"], timeout=15, max_bytes=30_000_000)
        row.update(info)
        with Image.open(ROOT / old["original_path"]) as im:
            im.load()
            row.update(width=im.width, height=im.height, format=im.format)
        row["sha256"] = hashlib.sha256((ROOT / old["original_path"]).read_bytes()).hexdigest()
        row["ok"] = True
        row.pop("error", None)
    except Exception as exc:
        row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:240]}")
    row["retry_seconds"] = round(time.monotonic() - t, 3)
    return row


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    logpath = SOURCE / "download_log.jsonl"
    recordpath = SOURCE / "acquired_records.json"
    retrylog = SOURCE / "download_retry_transient.jsonl"
    if retrylog.exists():
        raise RuntimeError("Retry log already exists: one-pass guard; inspect it before requesting another pass")
    rows = [json.loads(s) for s in logpath.read_text(encoding="utf-8").splitlines() if s.strip()]
    records = json.loads(recordpath.read_text(encoding="utf-8"))
    chosen = [r for r in rows if eligible(r)][:80]
    for original, snapshot in [(logpath, SOURCE / "download_log_initial.jsonl"),
                               (recordpath, SOURCE / "acquired_records_initial.json")]:
        if snapshot.exists():
            raise RuntimeError(f"Initial snapshot already exists; refusing to replace it: {snapshot}")
        shutil.copy2(original, snapshot)
    print(f"Eligible {sum(eligible(r) for r in rows)}; retry one pass {len(chosen)} URLs", flush=True)
    results = {}
    with retrylog.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(attempt, r) for r in chosen]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results[row["url"]] = row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"Retry {i}/{len(chosen)}; recovered {sum(r['ok'] for r in results.values())}", flush=True)
    new_successes = {url for url, row in results.items() if row.get("ok")}
    new_crops = 0
    for row in records:
        result = results.get(row["url"])
        if result is None:
            continue
        row.update(retried=True, attempts=2, download_ok=result["ok"])
        if row["url"] not in new_successes:
            row.update(crop_ok=False, error=result.get("error"))
            continue
        try:
            # Identical reconstruction to acquire_real.rof: clipped author box,
            # RGB PNG; no resize/alignment/augmentation or model preprocessing.
            with Image.open(ROOT / result["original_path"]) as im:
                im = im.convert("RGB")
                x, y, w, h = row["author_box_xy_width_height"]
                box = (max(0, x), max(0, y), min(im.width, x + w), min(im.height, y + h))
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("invalid_author_box")
                path = DATA / "benchmark_views/rof" / row["condition"] / row["identity"] / (row["image_id"] + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                im.crop(box).save(path)
                row.update(path=path.relative_to(ROOT).as_posix(), crop_ok=True,
                           clipped_box=list(box), box_clipped=list(box) != [x, y, x + w, y + h])
                row.pop("error", None)
                new_crops += 1
        except Exception as exc:
            row.update(crop_ok=False, error=str(exc))
    merged = [results.get(row["url"], row) for row in rows]
    temporary = logpath.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for row in merged:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(logpath)
    atomic_json(recordpath, records)
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "One pass; four workers; fetch timeout15; max80; exclude auth/rate-limit/DNS/TLS; HTTP525/526 excluded",
        "retry_urls": len(chosen), "new_valid_originals": len(new_successes),
        "new_valid_crops": new_crops,
        "current_valid_originals": sum(row.get("ok", False) for row in merged),
        "current_valid_crops": sum(row.get("crop_ok", False) for row in records),
        "unretried_rows_unchanged": all(a == b for a, b in zip(rows, merged) if a["url"] not in results),
        "initial_retry_errors": dict(Counter(row.get("error") for row in chosen)),
        "remaining_retry_errors": dict(Counter(row.get("error") for row in results.values() if not row.get("ok"))),
        "http429_deferred": sum("HTTP Error 429:" in row.get("error", "") for row in rows),
    }
    atomic_json(SOURCE / "retry_transient_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
