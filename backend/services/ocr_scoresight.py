# backend/services/ocr_scoresight.py
from __future__ import annotations
import os, time, re
from typing import Any, Dict, List, Optional, Tuple
import requests

"""
ScoreSight adapter (pull mode)

Expects ScoreSight's built-in Browser server to be running and serving JSON at:
  http://127.0.0.1:<port>/json

We normalize its fields into your pipeline schema:
{
  "away_team": str | None,
  "home_team": str | None,
  "score":     str | None,   # "AWAY-HOME" (e.g. "7-10")
  "clock":     str | None,   # "M:SS"
  "quarter":   str | None,   # "Q1".."Q4" or "OT"
  "used_stub": bool
}
"""

SCORESIGHT_URL = os.getenv("SCORESIGHT_JSON_URL", "http://127.0.0.1:18099/json")
HTTP_TIMEOUT_S = float(os.getenv("SCORESIGHT_HTTP_TIMEOUT_S", "1.5"))
POLL_ATTEMPTS  = int(os.getenv("SCORESIGHT_POLL_ATTEMPTS", "3"))
POLL_SLEEP_S   = float(os.getenv("SCORESIGHT_POLL_SLEEP_S", "0.05"))

# --- parsing helpers ---------------------------------------------------------

_SCORE_RX   = re.compile(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$")
_CLOCK_RX   = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$")
_QTR_RX_QN  = re.compile(r"^(Q[1-4]|OT)$", re.I)
_QTR_RX_MDN = re.compile(r"^(1ST|2ND|3RD|4TH|OT)$", re.I)

def _normalize_quarter(s: Optional[str]) -> Optional[str]:
    if not s: return None
    x = s.strip().upper()
    # common ScoreSight glitches like "111st" -> "1ST"
    x = re.sub(r"1{2,}", "1", x)
    x = x.replace("  ", "").replace(" ", "")
    x = x.replace("0T", "OT").replace("QI", "Q1").replace("QT", "Q1")
    m = _QTR_RX_QN.match(x)
    if m: return m.group(1)
    m2 = _QTR_RX_MDN.match(x)
    if m2: return {"1ST":"Q1","2ND":"Q2","3RD":"Q3","4TH":"Q4","OT":"OT"}[m2.group(1)]
    if re.fullmatch(r"[1-4]", x):  # "1" -> "Q1"
        return f"Q{x}"
    return None

def _normalize_clock(s: Optional[str]) -> Optional[str]:
    if not s: return None
    x = s.strip()
    # ignore obvious garbage like ". ." etc.
    if not re.search(r"\d", x):
        return None
    # normalize visually-broken separators
    x = x.replace(";", ":").replace(" ", "")
    m = _CLOCK_RX.match(x)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return None

def _digits_or_none(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = re.search(r"\d{1,2}", s)
    return m.group(0) if m else None

def _compose_score(away_s: Optional[str], home_s: Optional[str], joined: Optional[str]) -> Optional[str]:
    a = _digits_or_none(away_s)
    h = _digits_or_none(home_s)
    if a is not None and h is not None:
        return f"{int(a)}-{int(h)}"
    if joined:
        m = _SCORE_RX.match(joined.replace(":", "-"))
        if m:
            return f"{int(m.group(1))}-{int(m.group(2))}"
    return None

# --- fetch & normalize -------------------------------------------------------

def _fetch_scoresight(url: str) -> Optional[List[Dict[str, Any]]]:
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT_S)
        if r.status_code != 200:
            return None
        j = r.json()
        if isinstance(j, list):
            return j
        # some versions may return a dict with "items"/"zones"
        if isinstance(j, dict):
            key = next((k for k in ("items","zones","data") if isinstance(j.get(k), list)), None)
            return j.get(key) if key else None
        return None
    except Exception:
        return None

def _index_by_name(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        name = str(it.get("name","")).strip().lower()
        if name:
            out[name] = it
    return out

ALIASES = {
    "away_team":  ["away team","team_away","team left","left team","visitor","away"],
    "home_team":  ["home team","team_home","team right","right team","home"],
    "away_score": ["away score","score_away","left score","score left","visitor score"],
    "home_score": ["home score","score_home","right score","score right","home score"],
    "clock":      ["time","clock","game clock"],
    "quarter":    ["period","quarter","qtr"],
    "score":      ["score","scoreline","score text"],
}

def _find_text(idx: Dict[str, Dict[str, Any]], keys: List[str]) -> Optional[str]:
    for k in keys:
        it = idx.get(k)
        if it:
            txt = it.get("text")
            if isinstance(txt, (int, float)): return str(int(txt))
            if isinstance(txt, str) and txt.strip() != "": return txt.strip()
    return None

def extract_scoreboard_via_scoresight(
    *,
    url: str = SCORESIGHT_URL,
    poll_attempts: int = POLL_ATTEMPTS,
    poll_sleep_s: float = POLL_SLEEP_S,
) -> Dict[str, Any]:
    """
    Poll the ScoreSight local server a few times and return the most complete sample.
    """
    best: Dict[str, Any] = {}
    def completeness(d: Dict[str, Any]) -> int:
        return sum(1 for k in ("away_team","home_team","score","clock","quarter") if d.get(k))

    for _ in range(max(1, poll_attempts)):
        items = _fetch_scoresight(url)
        if not items:
            time.sleep(poll_sleep_s)
            continue

        idx = _index_by_name(items)

        away_team = _find_text(idx, ALIASES["away_team"])
        home_team = _find_text(idx, ALIASES["home_team"])
        away_sc   = _find_text(idx, ALIASES["away_score"])
        home_sc   = _find_text(idx, ALIASES["home_score"])
        clock     = _find_text(idx, ALIASES["clock"])
        period    = _find_text(idx, ALIASES["quarter"])
        joined    = _find_text(idx, ALIASES["score"])

        q_norm = _normalize_quarter(period)
        c_norm = _normalize_clock(clock)
        s_norm = _compose_score(away_sc, home_sc, joined)

        cur = {
            "away_team": away_team,
            "home_team": home_team,
            "score": s_norm,
            "clock": c_norm,
            "quarter": q_norm,
        }

        if completeness(cur) > completeness(best):
            best = cur
            if completeness(best) == 5:
                break

        time.sleep(poll_sleep_s)

    if not best:
        return {"used_stub": True}
    best["used_stub"] = False
    return best
