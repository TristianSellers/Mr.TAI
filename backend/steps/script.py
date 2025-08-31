def segments_to_commentary(segments: list[dict], max_lines: int = 3) -> str:
    """
    Heuristic: take the first few non-empty lines, trim, join with pauses.
    """
    lines = []
    for s in segments:
        t = (s.get("text") or "").strip()
        if t:
            lines.append(t)
        if len(lines) >= max_lines:
            break
    if not lines:
        lines = ["Play starts now.", "Nice move!", "What a finish!"]
    # add simple pacing cues; pyttsx3 respects punctuation pauses
    return " ... ".join(lines)
