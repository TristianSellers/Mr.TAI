from pathlib import Path
import pyttsx3

def synth_to_wav(text: str, out_path: Path, rate: int = 200) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    # Optional tweaks:
    engine.setProperty("rate", rate)
    # engine.setProperty("voice", "com.apple.speech.synthesis.voice.Alex")  # example mac voice
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    return out_path
