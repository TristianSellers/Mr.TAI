from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
import uuid, time, os

# Respect the same DATA_DIR your app uses
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

class JobState(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"

@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    state: JobState = JobState.queued
    error: Optional[str] = None
    # optional: store original filename you copied in
    input_name: Optional[str] = None

    def dir(self) -> Path:
        return DATA_DIR / "jobs" / self.id

    def ensure_layout(self) -> None:
        for sub in ["input", "frames", "events", "audio", "outputs"]:
            (self.dir() / sub).mkdir(parents=True, exist_ok=True)

    def to_api(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        d["root"] = str(self.dir().resolve())
        return d
