from fastapi import APIRouter
from backend.main import DATA_DIR, to_static_url
from pathlib import Path

router = APIRouter()

@router.get("/demo/artifacts")
def demo_artifacts():
    demo_dir = DATA_DIR / "demo"
    return {
        "text": (demo_dir / "demo_text.txt").read_text(encoding="utf-8") if (demo_dir / "demo_text.txt").exists() else None,
        "audio_url": to_static_url(demo_dir / "demo_commentary.mp3"),
        "video_url": to_static_url(demo_dir / "demo_dubbed.mp4"),
        "clip_url": to_static_url(demo_dir / "demo_clip.mp4"),
    }
