from dataclasses import dataclass, field
from typing import List, Dict, Any, Union


LABELS = [
"run", "pass_attempt", "completion", "incomplete",
"sack", "qb_scramble", "interception", "fumble",
"touchdown", "tackle_big_hit"
]


@dataclass
class Event:
    t_start: float
    t_end: float
    primary_label: str
    confidence: float
    alt_labels: List[Dict[str, Union[str, float]]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameplayPrediction:
    clip_id: str
    events: List[Event]


@dataclass
class BannerFacts:
    text: str = ""
    yard_gain: int | None = None
    result: str | None = None # e.g., "first_down"


@dataclass
class ScoreboardFacts:
    qtr: int | None = None
    clock: str | None = None
    down: int | None = None
    distance: int | None = None
    yardline: str | None = None
    score: Dict[str, int] | None = None