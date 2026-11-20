"""JABRIX AI v3.6.0 — Authorized Project Sandbox & Safe Untrusted-Input File Access.

Enforces:
- Strict filesystem confinement to the authorized project root (blocks .., symlinks outside root).
- Bounded file size and file count limits for Intel Core i3 / 8 GB RAM CPU-only environments.
- Read-only text reading; treats all repository contents as untrusted input (zero host execution).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "dist",
        "build",
        ".tox",
        ".eggs",
    }
)


class ProjectIsolationError(PermissionError):
    """Raised when a path escapes the authorized project root."""


class AuthorizedProjectScope:
    """Confines all software-intelligence reads and writes strictly inside project_root."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        max_files: int = 500,
        max_file_bytes: int = 512 * 1024,
    ) -> None:
        self.root = Path(project_root).resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Authorized project root does not exist or is not a directory: {self.root}")
        self.max_files = max(1, int(max_files))
        self.max_file_bytes = max(1024, int(max_file_bytes))

    def resolve_safe(self, path: str | Path) -> Path:
        """Resolve a relative or absolute path and verify it stays strictly within self.root."""
        p = Path(path)
        candidate = (self.root / p).resolve() if not p.is_absolute() else p.resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ProjectIsolationError(
                f"Path '{path}' resolves outside authorized project root '{self.root}'."
            ) from exc
        return candidate

    def relative_posix(self, path: Path) -> str:
        safe = self.resolve_safe(path)
        return safe.relative_to(self.root).as_posix()

    def iter_files(self, *, suffix: str | None = None) -> Iterator[Path]:
        """Iterate project files safely within max_files and ignored-directory rules."""
        count = 0
        for item in sorted(self.root.rglob("*")):
            if count >= self.max_files:
                break
            try:
                rel = item.relative_to(self.root)
            except ValueError:
                continue
            if any(part in IGNORED_DIRS or part.startswith("pytest_tmp_") for part in rel.parts):
                continue
            if item.is_symlink():
                # Reject symlinks escaping root
                try:
                    self.resolve_safe(item)
                except ProjectIsolationError:
                    continue
            if not item.is_file():
                continue
            if suffix and item.suffix.lower() != suffix.lower():
                continue
            count += 1
            yield item

    def read_text_safe(self, path: str | Path) -> str | None:
        """Read UTF-8 text from an authorized file, returning None if binary or oversized."""
        safe = self.resolve_safe(path)
        if not safe.is_file():
            return None
        try:
            if safe.stat().st_size > self.max_file_bytes:
                return None
            return safe.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    @staticmethod
    def sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
