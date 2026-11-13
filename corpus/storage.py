"""Safe corpus paths, immutable raw storage, and atomic processed outputs."""
from pathlib import Path
import os
import tempfile


class StorageManager:
    def __init__(self, project_root):
        self.root = Path(project_root).resolve()

    def resolve(self, relative):
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("storage path must remain under the project root") from exc
        return path

    def create_new(self, relative, mode="wb"):
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing corpus file: {path}")
        return path.open(mode)

    def atomic_jsonl(self, relative, rows):
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing corpus file: {path}")
        fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    import json
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise FileExistsError(f"destination appeared concurrently: {path}")
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return path
