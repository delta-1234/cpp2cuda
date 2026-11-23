"""Offline page retrieval from the pinned NVIDIA manuals."""
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/"offline_docs/nvidia"
TOKEN = re.compile(r"[a-z0-9_]+")


class Library:
    def __init__(self, root=ROOT):
        self.root = root
        self.docs = {d["id"]: d for d in json.loads((root/"manifest.json").read_text())["documents"]}

    def catalog(self):
        return [{k: d[k] for k in ("id", "title", "version", "pages")} for d in self.docs.values()]

    @lru_cache(maxsize=8)
    def pages(self, doc):
        if doc not in self.docs:
            raise ValueError("unknown document; use catalog")
        entry = self.docs[doc]
        raw = gzip.decompress((self.root/entry["text"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != entry["text_sha256"]:
            raise ValueError("offline document hash mismatch: "+doc)
        return raw.decode().rstrip("\f").split("\f")

    def read(self, doc, page, offset=0):
        pages = self.pages(doc)
        if type(page) != int or not 1 <= page <= len(pages) or type(offset) != int or offset < 0:
            raise ValueError("page is 1-based; offset is a nonnegative character index")
        body = pages[page-1]
        return {"doc":doc, "version":self.docs[doc]["version"], "page":page, "offset":offset,
                "text":body[offset:offset+4500], "next_offset":offset+4500 if len(body)>offset+4500 else None}

    def search(self, query, doc=None):
        terms = set(TOKEN.findall(query.lower()))
        if not terms:
            raise ValueError("use English terms or CUDA identifiers from the manuals")
        hits = []
        for key in ([doc] if doc else self.docs):
            pages = self.pages(key)
            counts = [Counter(t for t in TOKEN.findall(page.lower()) if t in terms) for page in pages]
            weights = {t: math.log(1 + len(pages)/(1+sum(t in c for c in counts))) for t in terms}
            for i,c in enumerate(counts):
                if c:
                    score = sum(weights[t]*n/(n+2) for t,n in c.items()) * len(c)/len(terms)
                    line = max(pages[i].splitlines(), key=lambda l:len(terms.intersection(TOKEN.findall(l.lower()))))
                    hits.append((score, key, i+1, line.strip()[:230]))
        return [{"doc":k,"version":self.docs[k]["version"],"page":p,"excerpt":line}
                for _,k,p,line in sorted(hits,reverse=True)[:6]]
