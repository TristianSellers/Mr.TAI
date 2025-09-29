# backend/services/mux.py
from pathlib import Path
import subprocess, shlex
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

    cmd = (
        f'ffmpeg -y -i {shlex.quote(str(video_path))} -i {shlex.quote(str(audio_path))} '
        f'-c:v copy -c:a aac -ar 48000 -ac 2 -b:a 128k -map 0:v:0 -map 1:a:0 -shortest '
        f'{shlex.quote(str(out_path))}'
    )
    subprocess.run(shlex.split(cmd), check=True)
    return out_path
