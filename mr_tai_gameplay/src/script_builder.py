from __future__ import annotations
from typing import List, Dict, Any, Optional
from .templates import one_liner

# -----------------------------
# Pacing / filler configuration
# -----------------------------
# Insert a filler when gap between events is >= this many seconds
GAP_MIN_FILLER = 2.0
# If the gap is very long, add multiple fillers spaced about every ~this seconds
GAP_MULTI_SPACING = 3.5
# Small nudge to keep t_say strictly increasing (avoid TTS overlap)
EPS_TIME = 0.12

FILLER = [
    "They’re regrouping.",
    "Setting up for the next snap.",
    "Crowd is getting into it.",
    "Coaches signaling in the play.",
    "Offense hurrying to the line.",
]

def _pick_filler(ix: int) -> str:
    return FILLER[ix % len(FILLER)]

def _insert_gap_fillers(lines: List[Dict[str, Any]], start_t: float, end_t: float, filler_ix: int) -> int:
    """
    Insert one or more filler lines inside the gap (start_t→end_t).
    Returns updated filler index.
    """
    gap = max(0.0, end_t - start_t)
    if gap < GAP_MIN_FILLER:
        return filler_ix

    # For long gaps, lay down multiple fillers spaced ~GAP_MULTI_SPACING apart
    num_fillers = max(1, int(gap // GAP_MULTI_SPACING))
    # If only one filler, place it mid-gap; otherwise, space them
    if num_fillers == 1:
        t = start_t + gap / 2.0
        lines.append({"t_say": round(t, 2), "text": _pick_filler(filler_ix)})
        filler_ix += 1
    else:
        step = gap / (num_fillers + 1)
        for i in range(1, num_fillers + 1):
            t = start_t + i * step
            lines.append({"t_say": round(t, 2), "text": _pick_filler(filler_ix)})
            filler_ix += 1
    return filler_ix

def _ensure_monotonic_times(lines: List[Dict[str, Any]]) -> None:
    """
    Make sure each t_say is strictly increasing by at least EPS_TIME.
    Adjust in-place with tiny nudges to avoid overlaps.
    """
    lines.sort(key=lambda x: x["t_say"])
    for i in range(1, len(lines)):
        prev_t = lines[i-1]["t_say"]
        if lines[i]["t_say"] <= prev_t:
            lines[i]["t_say"] = round(prev_t + EPS_TIME, 2)

def build_script(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a time-aware commentary script given event windows.

    events: list of dicts like asdict(Event) (must include t_start, t_end, primary_label)
    Returns: {"lines":[{"t_say":float,"text":str}, ...]}

    Rules:
      - Speak on each event end (t_end + 0.30).
      - If there is a gap >= GAP_MIN_FILLER between events, add filler at mid-gap (or multiple for long gaps).
      - Skip event line when primary_label == "no_event" (still allow gap fillers).
      - Ensure returned lines are sorted and strictly increasing in time.
    """
    lines: List[Dict[str, Any]] = []
    filler_ix = 0

    # Sort by start time and ignore any malformed entries early
    evs = [
        ev for ev in events
        if "t_start" in ev and "t_end" in ev and "primary_label" in ev
    ]
    evs.sort(key=lambda ev: float(ev["t_start"]))

    prev_end: Optional[float] = None

    for ev in evs:
        label = str(ev["primary_label"])
        t_start = float(ev["t_start"])
        t_end = float(ev["t_end"])

        # Gap fillers between previous event end and current event start
        if prev_end is not None:
            filler_ix = _insert_gap_fillers(lines, prev_end, t_start, filler_ix)

        # Event line: use templates; future: pass scoreboard/banner per-window if available
        if label != "no_event":
            text = one_liner(label, None, None, None).strip()
            if text:
                lines.append({"t_say": round(t_end + 0.30, 2), "text": text})

        prev_end = t_end

    # No events? Return empty payload
    if not lines:
        return {"lines": []}

    # Enforce monotonic timing to avoid overlaps/duplicates
    _ensure_monotonic_times(lines)
    return {"lines": lines}
