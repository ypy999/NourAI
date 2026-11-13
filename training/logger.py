import json
import os
from pathlib import Path
import threading
import time


class JsonlLogger:
    """Append-only, flushed JSONL event log."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event):
        row = {"timestamp": time.time(), **event}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
