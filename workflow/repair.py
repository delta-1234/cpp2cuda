"""Track repeated failures and preserve independently verified CUDA source revisions."""
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path


class RepairHistory:
    def __init__(self, folder, sources):
        self.folder = folder/"checkpoints"
        self.sources = sources
        self.failures = []
        self.checkpoints = {}
        self.recent = []

    def files(self):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.sources.iterdir())}

    def checkpoint(self):
        files = self.files()
        key = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
        if key not in self.checkpoints:
            self.folder.mkdir(exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".new-",dir=self.folder))
            try:
                for name in files:
                    shutil.copyfile(self.sources/name,stage/name)
                stage.rename(self.folder/key)
            finally:
                if stage.exists(): shutil.rmtree(stage)
            self.checkpoints[key] = {"source_sha256":files}
        return key

    def restore(self, key):
        if key not in self.checkpoints:
            raise ValueError("choose a verified checkpoint ID from diagnostics")
        staged = Path(tempfile.mkdtemp(prefix="restore-",dir=self.sources.parent.parent))
        expected = self.checkpoints[key]["source_sha256"]
        try:
            for name,digest in expected.items():
                data = (self.folder/key/name).read_bytes()
                if hashlib.sha256(data).hexdigest()!=digest: raise ValueError("checkpoint hash mismatch")
                (staged/name).write_bytes(data)
        except (OSError,ValueError):
            shutil.rmtree(staged)
            raise
        # Keep the replaced draft outside the build directory for later inspection.
        previous = staged.with_name(staged.name+"-previous")
        self.sources.rename(previous)
        try:
            staged.rename(self.sources)
        except OSError:
            previous.rename(self.sources)
            raise
        return previous.name

    def failure(self, tool, diagnostic, target=None, log=None):
        # Ignore transient compilation filenames and source line offsets when tracking the same diagnostic.
        error = next((line for line in diagnostic.splitlines() if re.search(r'(?:fatal )?error[: #]',line,re.I)),None)
        signature = error or diagnostic.split("\n",1)[0]
        signature = re.sub(r"tmpxft_[\w.]+|:\d+(?::\d+)?(?=:)",":*",signature)
        token = (tool,repr(target),signature)
        self.recent.append(token)
        self.recent = self.recent[-8:]
        prior = [f for f in self.failures if f["status"]=="open" and (f["tool"],f["target"],f["signature"])==(tool,target,signature)]
        item = {"id":len(self.failures)+1,"tool":tool,"target":target,"signature":signature,
                "count":len(prior)+1,"status":"open","diagnostic":diagnostic,"log":log}
        seq = self.recent
        if len(seq)>=4 and seq[-4:-2]==seq[-2:] and seq[-1]!=seq[-2]:
            item["loop"] = "alternating_failures"
        elif prior:
            item["loop"] = "repeated_failure"
        self.failures.append(item)
        return item

    def passed(self, tool, target=None):
        for failure in self.failures:
            if failure["status"]!="open": continue
            if ((failure["tool"]==tool and (tool!="test_case" or failure["target"]==target))
                    or (tool=="verify" and failure["tool"]=="test_case")):
                failure["status"]="passed_on_retry"
        if tool=="verify": self.recent.clear()
