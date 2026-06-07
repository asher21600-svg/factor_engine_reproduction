#!/usr/bin/env python
"""Acquire Qlib-format CN A-share daily data — WITHOUT needing the qlib package.

Downloads a prebuilt Qlib `.bin` bundle (default: the actively-maintained
community bundle `chenditc/investment_data`, current through the latest release)
and extracts it to ~/.qlib/qlib_data/cn_data, where `fe.data.qlib_bin` (a pure-
NumPy reader) and the rest of the pipeline can use it.  No C compiler / no qlib
install required.

    python data/setup_qlib.py                 # ~1.5 GB download, extracts to default dir
    python data/setup_qlib.py --target /data/cn_data --force
    QLIB_DATA_URL=<url> python data/setup_qlib.py     # custom mirror

Fallback: if a direct download fails but the `qlib` package is importable, uses
qlib's own GetData downloader.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get(
    "QLIB_DATA_URL",
    "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz")

# project root on path for the verifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _content_length(url: str) -> int:
    """Total size via a Range probe (handles GitHub release CDN redirects)."""
    req = urllib.request.Request(url, headers={"User-Agent": "factor-engine-repro",
                                               "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1])
            return int(r.headers.get("Content-Length", 0))
    except Exception:  # noqa: BLE001
        return 0


def _download(url: str, dst: Path, max_retries: int = 10):
    """Resumable, retrying download to `dst` (persistent cache → survives drops)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = _content_length(url)
    print(f"  downloading {url}  (total {total/1e6:.0f} MB)" if total else f"  downloading {url}")
    next_mark = 0
    for attempt in range(1, max_retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if total and have >= total:
            break
        headers = {"User-Agent": "factor-engine-repro"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have else "wb"
                if have and getattr(r, "status", 206) == 200:   # server ignored Range
                    mode, have = "wb", 0
                with open(dst, mode) as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        if have >= next_mark:
                            pct = (100 * have / total) if total else 0
                            print(f"    {have/1e6:8.1f} MB"
                                  + (f" / {total/1e6:.0f} MB ({pct:4.1f}%)" if total else ""),
                                  flush=True)
                            next_mark = have + (50 << 20)
        except Exception as e:  # noqa: BLE001
            got = dst.stat().st_size if dst.exists() else 0
            print(f"    [drop at {got/1e6:.0f} MB; retry {attempt}/{max_retries}] "
                  f"{type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        if total and dst.stat().st_size >= total:
            break
        print(f"    [stream ended early at {dst.stat().st_size/1e6:.0f} MB; resuming]", flush=True)
    final = dst.stat().st_size if dst.exists() else 0
    if total and final < total:
        raise IOError(f"incomplete download: {final}/{total} bytes after {max_retries} retries")
    print(f"  downloaded {final/1e6:.1f} MB")


def _find_bundle_root(extract_dir: Path) -> Path | None:
    """Locate the directory that contains calendars/day.txt within the archive."""
    for cal in extract_dir.rglob("calendars/day.txt"):
        return cal.parent.parent
    return None


def _via_download(url: str, target: Path) -> bool:
    import shutil
    # persistent cache so a dropped/partial download resumes across retries & re-runs
    cache = target.parent / "_download"
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / "qlib_bin.tar.gz"
    try:
        _download(url, tgz)
    except Exception as e:  # noqa: BLE001
        print(f"  download failed: {type(e).__name__}: {e}")
        print(f"  (partial saved at {tgz} — rerun to resume)")
        return False
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print("  extracting ...")
        try:
            with tarfile.open(tgz, "r:*") as tf:
                tf.extractall(td)
        except Exception as e:  # noqa: BLE001
            print(f"  extract failed ({e}); deleting cache so next run re-downloads")
            tgz.unlink(missing_ok=True)
            return False
        root = _find_bundle_root(td)
        if root is None:
            print("  archive has no calendars/day.txt — unexpected layout")
            return False
        target.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            dest = target / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
    tgz.unlink(missing_ok=True)            # success → free the ~409 MB cache
    return True


def _via_qlib(target: Path) -> bool:
    try:
        from qlib.tests.data import GetData
    except Exception as e:  # noqa: BLE001
        print(f"  qlib package not available for fallback: {e}")
        return False
    try:
        GetData().qlib_data(target_dir=str(target), region="cn", exists_skip=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  qlib GetData failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.path.expanduser("~/.qlib/qlib_data/cn_data"))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    target = Path(args.target).expanduser()

    from fe.data import qlib_bin
    if qlib_bin.available(str(target)) and not args.force:
        cal = qlib_bin.read_calendar(str(target))
        print(f"[ok] data already present at {target} "
              f"({len(cal)} trading days, {cal.min().date()}..{cal.max().date()})")
        print("    (use --force to re-download)")
        return

    print(f"== Qlib CN data setup -> {target} ==")
    ok = _via_download(args.url, target)
    if not ok:
        print("  direct download failed; trying qlib package fallback ...")
        ok = _via_qlib(target)
    if not ok:
        print("\n[FAIL] Could not acquire data. Options:")
        print("  - set QLIB_DATA_URL to a reachable prebuilt Qlib .bin bundle, or")
        print("  - install qlib on a machine with a C toolchain and rerun, or")
        print("  - build from a vendor (AkShare/baostock/Tushare) -> Qlib dump_bin.")
        sys.exit(1)

    if qlib_bin.available(str(target)):
        cal = qlib_bin.read_calendar(str(target))
        print(f"\n[ok] ready: {len(cal)} trading days, {cal.min().date()}..{cal.max().date()}")
        print(f"     verify with:  python tests/test_data.py")
    else:
        print("\n[FAIL] extracted but no readable bundle found at target")
        sys.exit(1)


if __name__ == "__main__":
    main()
