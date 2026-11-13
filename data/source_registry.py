"""Runtime source registry loaded from configs/dataset_sources.json."""
import json
from pathlib import Path


class SourceRegistry:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "dataset_sources.json"
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("dataset source registry must be a JSON object")
        self.sources = value

    def list_sources(self):
        return [{"source_id": source_id, **metadata} for source_id, metadata in self.sources.items()]

    def get(self, source_id):
        try:
            return {"source_id": source_id, **self.sources[source_id]}
        except KeyError as exc:
            raise ValueError(f"unknown source id: {source_id}") from exc

    def verified_for_ingestion(self, source_id):
        source = self.get(source_id)
        return source["status"] == "VERIFIED" and source.get("approved_for_ingestion") is True
