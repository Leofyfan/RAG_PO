from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

PHEME_FIGSHARE_URL = "https://ndownloader.figshare.com/files/6453753"


def download_pheme_archive(out_path: str | Path, url: str = PHEME_FIGSHARE_URL) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    tmp = out.with_suffix(out.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(out)
    return out


def verify_pheme_archive(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1_000_000:
        return False
    try:
        with tarfile.open(p, "r:*") as tf:
            return any(name.endswith(".json") for name in tf.getnames()[:5000])
    except tarfile.TarError:
        return False
