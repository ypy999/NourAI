import argparse,json,sqlite3
from pathlib import Path

def rollback(db_path,map_path):
    db=Path(db_path); rows=[json.loads(x) for x in Path(map_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    con=sqlite3.connect(str(db)); con.execute("PRAGMA foreign_keys=ON")
    try:
        con.execute("BEGIN IMMEDIATE")
        for row in rows:
            found=con.execute("SELECT metadata_json FROM records WHERE record_id=?",(row["record_id"],)).fetchone()
            if not found: raise RuntimeError(f"missing record_id {row['record_id']}")
            meta=json.loads(found[0]); marker=meta.get("source_record_id_migration",{})
            if marker.get("status")!="VERIFIED" or marker.get("previous_legacy_value")!=row["old_source_record_id"]: raise RuntimeError("migration marker mismatch; refusing rollback")
            meta.pop("source_record_id_migration",None)
            con.execute("UPDATE records SET source_record_id=?,metadata_json=? WHERE record_id=?",(row["old_source_record_id"],json.dumps(meta,ensure_ascii=False,sort_keys=True),row["record_id"]))
        con.commit()
    except Exception:
        con.rollback(); raise
    finally: con.close()
    con=sqlite3.connect(str(db))
    try:
        if con.execute("PRAGMA integrity_check").fetchall()!=[("ok",)]: raise RuntimeError("integrity_check failed after rollback")
        if con.execute("PRAGMA foreign_key_check").fetchall(): raise RuntimeError("foreign_key_check failed after rollback")
    finally: con.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--db",required=True); p.add_argument("--map",required=True); p.add_argument("--apply",action="store_true")
    a=p.parse_args()
    if not a.apply: print("DRY RUN: pass --apply only after creating a SQLite Backup API backup")
    else: rollback(a.db,a.map); print("ROLLBACK_OK")
