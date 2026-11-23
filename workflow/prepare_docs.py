"""Fetch official manuals during preparation, or index previously downloaded PDFs offline."""
import argparse
import datetime
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

DEST = Path(__file__).resolve().parents[1] / "offline_docs/nvidia"
SOURCES = DEST.parent / "sources.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(mode, dest):
    dest.mkdir(exist_ok=True, parents=True)
    manifest = []
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        staging = Path(tmp)
        seen = set()
        for source in json.loads(SOURCES.read_text()):
            key, title, url = source["id"], source["title"], source["url"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",key) or key in seen:
                raise ValueError("manual IDs must be unique filenames")
            seen.add(key)
            pdf = staging / (key + ".pdf")
            if mode != "index":
                with urlopen(url, timeout=90) as source, pdf.open("wb") as target:
                    shutil.copyfileobj(source, target)
            else:
                pdf = dest / pdf.name
            data = pdf.read_bytes()
            if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-1024:]:
                raise ValueError(f"incomplete PDF: {key}")
            if mode == "fetch":
                continue
            raw = subprocess.check_output(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"])
            pages = raw.decode().rstrip("\f").split("\f")
            if len(pages) < 2 or not any(p.strip() for p in pages):
                raise ValueError(f"empty manual: {key}")
            text = staging / (key + ".txt.gz")
            text.write_bytes(gzip.compress(raw, mtime=0))
            version = re.search(r'(?:Release\s+|Version\s+|_v)(\d+\.\d+(?:\.\d+)*)', pages[0], re.I)
            manifest.append(dict(id=key, title=title, version=source.get("version") or (version[1] if version else "see cover"),
                                 source=url, pdf=pdf.name, pdf_sha256=sha(data), text=text.name, text_sha256=sha(raw), pages=len(pages)))
        if manifest:
            (staging/"manifest.json").write_text(json.dumps({"indexed_utc":datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                                                              "documents":manifest}, indent=2)+"\n")
        for item in sorted(staging.iterdir(), key=lambda p: p.name=="manifest.json"):
            item.replace(dest / item.name)


def verify(dest):
    for entry in json.loads((dest/"manifest.json").read_text())["documents"]:
        pdf = (dest/entry["pdf"]).read_bytes()
        raw = gzip.decompress((dest/entry["text"]).read_bytes())
        count = len(raw.decode().rstrip("\f").split("\f"))
        if sha(pdf)!=entry["pdf_sha256"] or sha(raw)!=entry["text_sha256"] or count!=entry["pages"]:
            raise ValueError("offline snapshot mismatch: "+entry["id"])
        print(entry["id"], entry["version"], count, "pages verified")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fetch", "index", "sync", "verify"))
    mode = p.parse_args().mode
    if mode=="verify":
        verify(DEST)
    else:
        prepare(mode, DEST)
