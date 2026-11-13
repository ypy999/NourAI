"""SQLite-backed near-duplicate candidate detection; never deletes records."""
import hashlib
import json
import math
import re

from data.deduplication import normalize_duplicate_key


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _shingles(text):
    words = _WORD_RE.findall(normalize_duplicate_key(text))
    if len(words) < 3:
        return set(words)
    return {" ".join(words[index:index + 3]) for index in range(len(words) - 2)}


def _minhash_bands(shingles, *, permutations=16, bands=4):
    if permutations <= 0 or bands <= 0 or permutations % bands:
        raise ValueError("permutations must be positive and divisible by bands")
    if not shingles:
        return []
    minima = []
    for seed in range(permutations):
        minima.append(min(int.from_bytes(hashlib.blake2b(f"{seed}:{item}".encode("utf-8"), digest_size=8).digest(), "big")
                          for item in shingles))
    width = permutations // bands
    output = []
    for band in range(bands):
        values = minima[band * width:(band + 1) * width]
        digest = hashlib.sha256(",".join(str(value) for value in values).encode("ascii")).hexdigest()
        output.append((band, digest))
    return output


class NearDuplicateDetector:
    """Find likely similar records for review while retaining every input.

    LSH candidate generation can miss similar pairs (false negatives). Jaccard
    over word trigrams is a surface similarity, not semantic equivalence.
    """
    def __init__(self, connection, *, similarity_threshold=0.85, candidate_limit=200,
                 permutations=16, bands=4):
        if isinstance(similarity_threshold, bool) or not isinstance(similarity_threshold, (int, float)) or not math.isfinite(similarity_threshold) or not 0 < similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be finite and in (0,1]")
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit <= 0:
            raise ValueError("candidate_limit must be a positive integer")
        if permutations <= 0 or bands <= 0 or permutations % bands:
            raise ValueError("permutations must be positive and divisible by bands")
        self.connection = connection
        self.threshold = float(similarity_threshold)
        self.candidate_limit = candidate_limit
        self.permutations = permutations
        self.bands = bands
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS near_records(
                record_id TEXT PRIMARY KEY, source_id TEXT, document_id TEXT,
                content_hash TEXT NOT NULL, shingles_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS near_buckets(
                band INTEGER NOT NULL, bucket TEXT NOT NULL, record_id TEXT NOT NULL,
                PRIMARY KEY(band,bucket,record_id)
            );
            CREATE INDEX IF NOT EXISTS idx_near_bucket_lookup ON near_buckets(band,bucket);
            CREATE TABLE IF NOT EXISTS near_findings(
                record_id_a TEXT NOT NULL, record_id_b TEXT NOT NULL,
                similarity REAL NOT NULL, reason TEXT NOT NULL,
                decision TEXT NOT NULL, candidate_limit_hit INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(record_id_a,record_id_b)
            );
        """)

    def add(self, record):
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError("record must contain string text")
        text = record["text"]
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id is required; pass the canonical provenance ID")
        content_hash = record.get("content_hash") or hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()
        existing = self.connection.execute("SELECT content_hash FROM near_records WHERE record_id=?", (record_id,)).fetchone()
        if existing:
            if existing[0] != content_hash:
                raise ValueError("record_id is already indexed with different content_hash")
            return {"matches": [], "candidate_limit_hit": False, "empty_shingle_record": False,
                    "already_indexed": True}
        shingles = _shingles(text)
        if not shingles:
            return {"matches": [], "candidate_limit_hit": False, "empty_shingle_record": True}
        bands = _minhash_bands(shingles, permutations=self.permutations, bands=self.bands)
        candidate_ids = set()
        for band, bucket in bands:
            candidate_ids.update(row[0] for row in self.connection.execute(
                "SELECT record_id FROM near_buckets WHERE band=? AND bucket=? ORDER BY record_id LIMIT ?",
                (band, bucket, self.candidate_limit + 1)))
        candidate_limit_hit = len(candidate_ids) > self.candidate_limit
        candidates = sorted(candidate_ids)[:self.candidate_limit]
        matches = []
        for prior_id in candidates:
            prior = self.connection.execute("SELECT shingles_json FROM near_records WHERE record_id=?", (prior_id,)).fetchone()
            if prior is None:
                continue
            previous = set(json.loads(prior[0]))
            union = len(shingles | previous)
            similarity = len(shingles & previous) / union if union else 1.0
            if similarity >= self.threshold:
                left, right = sorted((prior_id, record_id))
                finding = {"record_id": record_id, "similar_record_id": prior_id,
                           "similarity": round(similarity, 6),
                           "classification": "probable_near_duplicate",
                           "reason": f"word-trigram Jaccard similarity >= configured threshold {self.threshold}",
                           "decision": "retain_both_pending_review"}
                self.connection.execute("INSERT OR IGNORE INTO near_findings VALUES(?,?,?,?,?,?)",
                    (left, right, similarity, finding["reason"], finding["decision"], int(candidate_limit_hit)))
                matches.append(finding)
        self.connection.execute("INSERT OR IGNORE INTO near_records VALUES(?,?,?,?,?)",
            (record_id, record.get("source_id"), record.get("document_id"), content_hash,
             json.dumps(sorted(shingles), ensure_ascii=False)))
        self.connection.executemany("INSERT OR IGNORE INTO near_buckets VALUES(?,?,?)",
                                    [(band, bucket, record_id) for band, bucket in bands])
        return {"matches": matches, "candidate_limit_hit": candidate_limit_hit,
                "empty_shingle_record": False}

    def findings(self):
        columns = ("record_id_a", "record_id_b", "similarity", "reason", "decision", "candidate_limit_hit")
        return [dict(zip(columns, row)) for row in self.connection.execute(
            "SELECT record_id_a,record_id_b,similarity,reason,decision,candidate_limit_hit "
            "FROM near_findings ORDER BY record_id_a,record_id_b")]

    def statistics(self):
        return {"records_indexed": self.connection.execute("SELECT COUNT(*) FROM near_records").fetchone()[0],
                "probable_pairs": self.connection.execute("SELECT COUNT(*) FROM near_findings").fetchone()[0],
                "similarity_metric": "Jaccard over normalized Unicode word trigrams",
                "similarity_threshold": self.threshold,
                "candidate_limit": self.candidate_limit,
                "records_deleted_or_excluded": 0,
                "limitations": ["LSH may miss similar pairs.", "Surface similarity is not semantic equivalence.",
                                "All flagged pairs are retained for review."]}
