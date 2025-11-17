from __future__ import annotations
import json
import subprocess
import tempfile
import wave
import zipfile
from pathlib import Path
from typing import Optional

import httpx
from vosk import KaldiRecognizer, Model


class LocalVoskTranscriber:
    """Offline transcription using Vosk + ffmpeg for media conversion."""

    def __init__(
        self,
        model_name: str,
        model_url: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cache_dir = cache_dir or Path(__file__).resolve().parents[1] / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self._ensure_model(model_name, model_url)
        self.model = Model(str(self.model_dir))

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        wav_path = self._convert_to_wav(audio_bytes, mime_type)
        try:
            with wave.open(str(wav_path), "rb") as wf:
                if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                    raise RuntimeError("Converted audio must be mono 16kHz.")
                rec = KaldiRecognizer(self.model, wf.getframerate())
                rec.SetWords(True)
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    rec.AcceptWaveform(data)
                result = json.loads(rec.FinalResult())
                return result.get("text", "").strip()
        finally:
            if wav_path.exists():
                wav_path.unlink(missing_ok=True)

    def _ensure_model(self, model_name: str, model_url: str) -> Path:
        target_dir = self.cache_dir / model_name
        if target_dir.exists():
            return target_dir
        zip_path = self.cache_dir / f"{model_name}.zip"
        if not zip_path.exists():
            self._download_model(model_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(self.cache_dir)
        if not target_dir.exists():
            raise RuntimeError(
                f"Extracted model directory {target_dir} not found. Please verify the zip contents."
            )
        return target_dir

    def _download_model(self, url: str, destination: Path) -> None:
        with httpx.stream("GET", url, timeout=60.0) as response:
            response.raise_for_status()
            with destination.open("wb") as file:
                for chunk in response.iter_bytes():
                    if chunk:
                        file.write(chunk)

    def _convert_to_wav(self, audio_bytes: bytes, mime_type: str) -> Path:
        ext = self._guess_extension(mime_type)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as src:
            src.write(audio_bytes)
            src_path = Path(src.name)
        dst_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        command = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst_path),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as err:
            raise RuntimeError(
                "ffmpeg is required for local transcription but was not found in PATH."
            ) from err
        except subprocess.CalledProcessError as err:
            raise RuntimeError("ffmpeg failed to convert audio chunk.") from err
        finally:
            src_path.unlink(missing_ok=True)
        return dst_path

    @staticmethod
    def _guess_extension(mime_type: str) -> str:
        mapping = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/opus": ".opus",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mp4": ".m4a",
        }
        return mapping.get(mime_type.lower(), ".bin")


