"""Audio extraction, Whisper transcription, and timeline alignment."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from openai import OpenAI


@dataclass(frozen=True)
class TranscriptSegment:
    """Whisper segment aligned to the video timeline."""

    start_second: float
    end_second: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


def extract_audio_track(video_path: str, output_dir: str, bitrate: str = "32k") -> Path:
    """Extract a Whisper-friendly mono mp3 track from the input video."""
    video = Path(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH")

    audio_path = out / f"{video.stem}.mp3"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        bitrate,
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def transcribe_audio_whisper(
    client: "OpenAI",
    audio_path: Path,
    output_path: Path,
    model: str = "whisper-1",
    language: str = "zh",
) -> Dict[str, Any]:
    """Transcribe full audio with Whisper and persist the raw response."""
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_or_create_transcript(
    video_path: str,
    output_dir: str,
    client: Optional["OpenAI"],
    model: str = "whisper-1",
    language: str = "zh",
) -> Dict[str, Any]:
    """Extract audio and return a cached or fresh Whisper transcription payload."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    transcript_path = out / "audio_transcript.json"
    if transcript_path.exists():
        return json.loads(transcript_path.read_text(encoding="utf-8"))

    if client is None:
        payload = {"text": "", "segments": [], "model": model, "language": language, "source": "empty-fallback"}
        transcript_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    audio_path = extract_audio_track(video_path=video_path, output_dir=str(out))
    return transcribe_audio_whisper(
        client=client,
        audio_path=audio_path,
        output_path=transcript_path,
        model=model,
        language=language,
    )


def parse_segments(transcript_payload: Dict[str, Any]) -> List[TranscriptSegment]:
    """Convert raw Whisper payload segments into typed timeline entries."""
    segments: List[TranscriptSegment] = []
    for segment in transcript_payload.get("segments", []):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_second=float(segment.get("start", 0.0)),
                end_second=float(segment.get("end", 0.0)),
                text=text,
                avg_logprob=segment.get("avg_logprob"),
                no_speech_prob=segment.get("no_speech_prob"),
            )
        )
    return segments


def transcript_window_text(
    segments: List[TranscriptSegment],
    center_second: float,
    window_before: float = 2.0,
    window_after: float = 2.0,
) -> str:
    """Return transcript text around a timestamp for frame-level prompting."""
    start = max(0.0, center_second - window_before)
    end = center_second + window_after
    texts = [seg.text for seg in segments if seg.end_second >= start and seg.start_second <= end]
    return " ".join(texts).strip()


def attach_transcript_to_events(
    events: List[Dict[str, Any]],
    segments: List[TranscriptSegment],
) -> List[Dict[str, Any]]:
    """Attach overlapping transcript snippets to each event."""
    enriched: List[Dict[str, Any]] = []
    for event in events:
        start = float(event.get("start_second", 0.0))
        end = float(event.get("end_second", start))
        matched_segments = [
            asdict(segment)
            for segment in segments
            if segment.end_second >= start and segment.start_second <= end
        ]
        enriched_event = dict(event)
        enriched_event["audio_context"] = {
            "transcript": " ".join(segment["text"] for segment in matched_segments).strip(),
            "segments": matched_segments,
        }
        enriched.append(enriched_event)
    return enriched
