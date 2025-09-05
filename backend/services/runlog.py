# backend/services/runlog.py
import json, datetime
from pathlib import Path
from backend.main import DATA_DIR

RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def append_run_log(entry: dict) -> None:
    date = datetime.datetime.utcnow().date().isoformat()
    path = RUNS_DIR / f"{date}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
