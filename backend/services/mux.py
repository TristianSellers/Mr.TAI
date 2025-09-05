# backend/services/mux.py
from pathlib import Path
import ffmpeg
from backend.main import DATA_DIR

DUB_DIR = (DATA_DIR / "uploads" / "dubbed")
DUB_DIR.mkdir(parents=True, exist_ok=True)

def mux_audio_video(video_path: str | Path, audio_path: str | Path, out_path: str | Path | None = None) -> Path:
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    if out_path is None:
        out_path = DUB_DIR / f"{video_path.stem}.dubbed.mp4"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy video, encode audio to AAC, truncate to shortest stream
    (
        ffmpeg
        .input(str(video_path))
        .output(str(audio_path))
    )  # no-op, but keeps parity if you later pre-process

    (
        ffmpeg
        .input(str(video_path))
        .input(str(audio_path))
        .output(
            str(out_path),
            **{
                "c:v": "copy",
                "c:a": "aac",
                "b:a": "128k",
                "shortest": None,  # flag
                "map": ["0:v:0", "1:a:0"],
                "loglevel": "error",
            },
        )
        .overwrite_output()
        .run()
    )
    return out_path
