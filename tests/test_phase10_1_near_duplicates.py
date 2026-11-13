import sqlite3

import pytest

from data.near_duplicates import NearDuplicateDetector


def test_near_duplicate_detection_logs_pairs_without_removal():
    db = sqlite3.connect(":memory:")
    detector = NearDuplicateDetector(db, similarity_threshold=0.6)
    first = {"record_id": "r1", "source_id": "s1",
             "text": "The quick brown fox jumps over the lazy dog and then runs down the road."}
    second = {"record_id": "r2", "source_id": "s2",
              "text": "The quick brown fox jumps over the lazy dog and then runs down a road today."}
    unrelated = {"record_id": "r3", "source_id": "s3",
                 "text": "Quantum mechanics describes wave functions and particle behavior."}
    assert detector.add(first)["matches"] == []
    matches = detector.add(second)["matches"]
    assert matches and matches[0]["similar_record_id"] == "r1"
    assert matches[0]["classification"] == "probable_near_duplicate"
    assert matches[0]["decision"] == "retain_both_pending_review"
    assert detector.add(unrelated)["matches"] == []
    assert detector.statistics()["records_deleted_or_excluded"] == 0
    assert len(detector.findings()) == 1
    assert detector.add(second)["already_indexed"]
    db.close()


def test_near_duplicate_threshold_and_stable_id_are_validated():
    db = sqlite3.connect(":memory:")
    with pytest.raises(ValueError):
        NearDuplicateDetector(db, similarity_threshold=1.5)
    detector = NearDuplicateDetector(db)
    with pytest.raises(ValueError):
        detector.add({"text": "record without canonical id"})
    db.close()
