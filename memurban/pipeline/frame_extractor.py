"""Frame extraction and deduplication for campus walk videos."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2


@dataclass(frozen=True)
class FrameSample:
    """Single extracted frame with stable timing metadata."""

    path: Path
    source_index: int
    timestamp_second: float


def extract_frames_ffmpeg(video_path: str, output_dir: str, fps: float = 0.5) -> List[Path]:
    """Extract frames with ffmpeg CLI; return sorted frame paths."""
    video = Path(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH")

    pattern = out / "frame_%05d.jpg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(out.glob("frame_*.jpg"))


def compute_hist_similarity(img1_path: Path, img2_path: Path) -> float:
    """Compute HSV histogram correlation similarity in [~ -1, 1]."""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return 0.0
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def deduplicate_frames(frame_paths: List[Path], threshold: float = 0.92) -> List[Path]:
    """Keep first frame and only keep subsequent frames with low similarity."""
    if not frame_paths:
        return []
    selected = [frame_paths[0]]
    for fp in frame_paths[1:]:
        sim = compute_hist_similarity(selected[-1], fp)
        if sim < threshold:
            selected.append(fp)
    return selected


def build_frame_samples(frame_paths: List[Path], fps: float) -> List[FrameSample]:
    """Attach stable timestamps to extracted frames based on extraction cadence."""
    if fps <= 0:
        raise ValueError("fps 必须大于 0")

    interval = 1.0 / fps
    return [
        FrameSample(
            path=frame_path,
            source_index=index,
            timestamp_second=round(index * interval, 3),
        )
        for index, frame_path in enumerate(frame_paths)
    ]


def extract_and_deduplicate(
    video_path: str,
    output_dir: str,
    fps: float = 0.5,
    similarity_threshold: float = 0.92,
) -> Tuple[List[FrameSample], List[FrameSample]]:
    """
    Run extraction + dedup.

    Returns:
        (all_frames, selected_frames)
    """
    all_dir = str(Path(output_dir) / "all")
    all_frame_paths = extract_frames_ffmpeg(video_path=video_path, output_dir=all_dir, fps=fps)
    all_frames = build_frame_samples(all_frame_paths, fps=fps)
    selected_paths = deduplicate_frames(all_frame_paths, threshold=similarity_threshold)
    selected = [sample for sample in all_frames if sample.path in set(selected_paths)]

    selected_dir = Path(output_dir) / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for fp in selected:
        target = selected_dir / fp.path.name
        if not target.exists():
            target.write_bytes(fp.path.read_bytes())
    return all_frames, selected
