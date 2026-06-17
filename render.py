"""ffmpeg tasks: extract audio for diarization/trancription and render the desired clip with captions."""

import subprocess

WIDTH, HEIGHT = 1080, 1920

# A 9:16 canvas captions burned into the video
# Both streams reset their timestamps to 0 so the muxed file has no edit-list
# offset, which otherwise makes some players drop the audio.
_FILTER = (
    "[0:v]split=2[bg][fg];"
    f"[bg]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
    f"crop={WIDTH}:{HEIGHT},gblur=sigma=24[bg];"
    f"[fg]scale={WIDTH}:-2[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2[v];"
    "[v]ass={ass},setpts=PTS-STARTPTS[outv];"
    "[0:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[outa]"
)


def _run(args: list[str]) -> None:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr[-1500:]}")


def extract_audio(src: str, out: str) -> None:
    """16 kHz mono mp3: all pyannote needs, and small to upload."""
    _run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
          "-c:a", "libmp3lame", "-q:a", "5", out])


def render(src: str, start: float, dur: float, ass: str, out: str) -> None:
    """Cut [start, start+dur] from src, reframe to vertical, and burn in the captions."""
    _run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
        "-filter_complex", _FILTER.format(ass=ass),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        out,
    ])
