import json
import shutil
from pathlib import Path

from corpus.pipeline import load_phase8_config, normalize_imported_file
from data.normalization import normalize_text, transform_record
from data.raw_storage import make_raw_record


def test_normalization_keeps_raw_and_records_each_text_stage(tmp_path):
    raw="\r\n  Bonjour\t le monde!  "
    row=make_raw_record(raw,source_id="tatoeba_cc0",license_id="CC0-1.0")
    transformed=transform_record(row)
    assert transformed["raw_text"]==raw
    assert transformed["normalized_text"]=="\n  Bonjour\t le monde!  "
    assert transformed["cleaned_text"]=="Bonjour le monde!"
    assert transformed["text"]==transformed["cleaned_text"]
    assert transformed["transformation_stages"]==["raw_text","normalized_text","cleaned_text"]
    assert normalize_text("e\u0301")=="é"


def test_normalize_imported_jsonl_persists_provenance_and_cleaned_text(tmp_path):
    (tmp_path/"configs").mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1]/"configs"/"default.json",tmp_path/"configs"/"default.json")
    imported=tmp_path/"data/processed/imported.jsonl"; imported.parent.mkdir(parents=True)
    raw="  bonjour\t tout le monde  "
    row=make_raw_record(raw,source_id="tatoeba_cc0",license_id="CC0-1.0"); row["text"]=raw
    imported.write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf-8")
    cfg=load_phase8_config()
    normalized=normalize_imported_file(imported,"data/processed/normalized.jsonl",project_root=tmp_path,config=cfg)
    cooked=json.loads(Path(normalized["output_file"]).read_text(encoding="utf-8"))
    assert cooked["raw_text"]==raw and cooked["normalized_text"]==raw
    assert cooked["cleaned_text"]=="bonjour tout le monde"
    assert cooked["raw_text_sha256"]==row["raw_text_sha256"]
