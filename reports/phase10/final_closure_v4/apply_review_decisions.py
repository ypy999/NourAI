import argparse,csv,json,shutil,hashlib
from datetime import datetime
from pathlib import Path
ROOT=Path(r"D:\\NourAI")
BASE=ROOT/"reports/phase10/final_closure_v2/splits_v2_verified"
OUT=ROOT/"reports/phase10/final_closure_v3"
CSV=OUT/"review_pairs.csv"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser(description="Apply human similarity decisions to a derived split copy only.")
 a.add_argument("--apply",action="store_true",help="write a new reviewed_splits copy; default is dry-run")
 z=a.parse_args()
 with CSV.open(encoding="utf-8-sig",newline="") as f: decisions=list(csv.DictReader(f))
 allowed={"","keep","drop_from_validation_test","merge"}; errors=[]; actions=[]
 for row in decisions:
  d=(row.get("decision") or "").strip().lower(); action=(row.get("action_record_id") or "").strip()
  if d not in allowed: errors.append(f"{row['id1']}/{row['id2']}: invalid decision {d}");continue
  if not d: continue
  if d=="keep": actions.append({"row":row,"decision":d});continue
  if not action or action not in {row["id1"],row["id2"]}: errors.append(f"{row['id1']}/{row['id2']}: action_record_id must be one of the pair");continue
  target_split=row["split1"] if action==row["id1"] else row["split2"]
  if d=="drop_from_validation_test" and target_split not in {"validation","test"}: errors.append(f"{action}: only validation/test records may be dropped");continue
  if d=="merge" and "UNMAPPED" in {row["split1"],row["split2"]}: errors.append(f"{row['id1']}/{row['id2']}: cannot merge an unmapped record into split copy");continue
  actions.append({"row":row,"decision":d,"action_record_id":action,"action_split":target_split})
 result={"mode":"APPLY" if z.apply else "DRY_RUN","pairs":len(decisions),"pending":sum(not (r.get("decision") or "").strip() for r in decisions),"validated_decisions":len(actions),"errors":errors,"writes_performed":False}
 if errors:
  print(json.dumps(result,ensure_ascii=False,indent=2)); raise SystemExit(2)
 if not z.apply or not actions:
  print(json.dumps(result,ensure_ascii=False,indent=2)); return
 target=OUT/"reviewed_splits"; target.mkdir(exist_ok=True)
 stamp=datetime.now().strftime("%Y%m%dT%H%M%S"); backup=target/"backups"/stamp; backup.mkdir(parents=True)
 for s in ("train","validation","test"):
  src=BASE/f"{s}.jsonl"
  if (target/f"{s}.jsonl").exists(): shutil.copy2(target/f"{s}.jsonl",backup/f"{s}.jsonl")
  else:
   shutil.copy2(src,backup/f"{s}.jsonl")
   shutil.copy2(src,target/f"{s}.jsonl")
 byid={}; items={s:[] for s in ("train","validation","test")}
 for s in items:
  with (target/f"{s}.jsonl").open(encoding="utf-8") as f:
   for line in f:
    obj=json.loads(line); items[s].append(obj); byid[obj.get("source_record_id")]=(s,obj)
 remove=set()
 for action in actions:
  if action["decision"]=="keep":continue
  row=action["row"]; chosen=action["action_record_id"]
  if chosen not in byid:raise ValueError(f"record id absent in split copy: {chosen}")
  if action["decision"]=="drop_from_validation_test":remove.add(chosen)
  elif action["decision"]=="merge":remove.add(row["id2"] if chosen==row["id1"] else row["id1"])
 for s in items:
  tmp=target/f"{s}.jsonl.tmp"
  with tmp.open("w",encoding="utf-8",newline="\n") as f:
   for obj in items[s]:
    if obj.get("source_record_id") not in remove:f.write(json.dumps(obj,ensure_ascii=False)+"\n")
  tmp.replace(target/f"{s}.jsonl")
 audit={"created_at":datetime.now().astimezone().isoformat(),"decisions":actions,"removed_ids":sorted(remove),"backup_dir":str(backup),"original_splits_modified":False,"merge_semantics":"retain action_record_id as survivor; remove the other split row; no text concatenation"}
 (target/"review_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 result["writes_performed"]=True
 result["removed_records"]=len(remove)
 result["reviewed_splits"]=str(target)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
