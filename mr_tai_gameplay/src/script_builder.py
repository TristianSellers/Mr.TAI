from __future__ import annotations
from typing import List, Dict, Any
from .templates import one_liner

# Light filler used when gaps are long enough
FILLER = [
    "They’re regrouping.",
    "Setting up for the next snap.",
    "Crowd is getting into it.",
]

def _pick_filler(ix: int) -> str:
    return FILLER[ix % len(FILLER)]

def build_script(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    events: list of dicts like asdict(Event) (already timed)
    Returns: {"lines":[{"t_say":float,"text":str}, ...]}

    Rules:
      - speak on each event end (t_end + 0.30)
      - if there is a gap >= 2.0s between events, insert a pause/filler at mid-gap
      - if event text is empty (e.g., no_event), skip the line
      - lines are returned sorted by t_say
    """
    lines: List[Dict[str, Any]] = []
    prev_end: float | None = None
    filler_ix = 0

    # Sort by start time just in case
    events_sorted = sorted(events, key=lambda ev: float(ev["t_start"]))

    for ev in events_sorted:
        label = ev["primary_label"]
        t_start = float(ev["t_start"])
        t_end = float(ev["t_end"])

        # Insert filler for long gaps between events
        if prev_end is not None:
            gap = max(0.0, t_start - prev_end)
            if gap >= 2.0:
                mid = prev_end + gap / 2.0
                lines.append({"t_say": round(mid, 2), "text": _pick_filler(filler_ix)})
                filler_ix += 1

        # Event line (use templates; yards/scoreboard can be threaded later)
        text = one_liner(label, None, None, None).strip()
        if text:
            lines.append({"t_say": round(t_end + 0.30, 2), "text": text})

        prev_end = t_end

    # Chronological order
    lines.sort(key=lambda x: x["t_say"])
    return {"lines": lines}
