"""Disk-backed exact, normalized, and SHA-256 duplicate detection."""
import hashlib
import unicodedata


def normalize_duplicate_key(text):
    """Return the documented normalized duplicate key.

    This profile uses NFC, Unicode casefolding, and whitespace collapsing. It
    preserves punctuation and diacritics; it is not semantic near-duplicate
    detection.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


class DiskDeduplicator:
    """SQLite-backed deduplication helper owned by the caller's transaction."""

    def __init__(self, connection):
        self.connection = connection
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen_records ("
            "exact_hash TEXT PRIMARY KEY, normalized_hash TEXT UNIQUE NOT NULL, "
            "exact_text TEXT NOT NULL, normalized_text TEXT NOT NULL, "
            "source TEXT NOT NULL, line_number INTEGER NOT NULL)"
        )

    @staticmethod
    def _digest(text):
        return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()

    def add_if_unique(self, record):
        """Return None for a new row, otherwise 'exact' or 'normalized'."""
        text = record["text"]
        normalized = normalize_duplicate_key(text)
        exact_hash = self._digest(text)
        normalized_hash = self._digest(normalized)
        exact = self.connection.execute(
            "SELECT exact_text FROM seen_records WHERE exact_hash = ?", (exact_hash,)
        ).fetchone()
        if exact is not None:
            if exact[0] != text:
                raise RuntimeError("SHA-256 collision detected for exact text")
            return "exact"
        normalized_row = self.connection.execute(
            "SELECT normalized_text FROM seen_records WHERE normalized_hash = ?", (normalized_hash,)
        ).fetchone()
        if normalized_row is not None:
            if normalized_row[0] != normalized:
                raise RuntimeError("SHA-256 collision detected for normalized text")
            return "normalized"
        self.connection.execute(
            "INSERT INTO seen_records(exact_hash, normalized_hash, exact_text, normalized_text, source, line_number) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (exact_hash, normalized_hash, text, normalized, record["source"], record["line_number"]),
        )
        return None
