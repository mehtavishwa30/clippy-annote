"""Gemini picks a clip for Youtube shorts generator from a diarized transcript.

The clip should contain 2 or more speakers. If Gemini returns a single-speaker
window we re-ask once, then fallback to the densest exchange in the transcript.
An optional `focus` steers the pick toward a specific topic or question.
"""

import json

from google import genai

MODEL = "gemini-2.5-flash"

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "start": {"type": "NUMBER", "description": "Clip start in seconds."},
        "end": {"type": "NUMBER", "description": "Clip end in seconds."},
        "title": {"type": "STRING", "description": "A punchy 3-7 word title for the clip."},
    },
    "required": ["start", "end", "title"],
}

PROMPT = """\
Here is a speaker-labelled transcript of a video, one turn per line as [start-end] SPEAKER: text.

{task} Requirements, in order:
- It MUST contain a real exchange between at least two different speakers — a back-and-forth such as a
  question and its answer, or two people building on each other. The speaker has to change at least once
  inside the window. Never pick a single-speaker monologue. the short relies on showing multiple speakers
  to demonstrate speaker diarization.
- It carries a genuinely interesting idea, not small talk or filler.
- It stands on its own without the surrounding context, opens on a strong line rather than mid-thought,
  and ends on a natural beat.

Snap start and end to turn boundaries. The window must be between {lo} and {hi} seconds long.
{transcript}"""

AUTO_TASK = "Pick the most insightful {lo}-{hi} second clip."
FOCUS_TASK = (
    'Find the {lo}-{hi} second window that best captures this specific topic or exchange:\n'
    '  "{focus}"\n'
    "Pick the part of the conversation that is actually about it and stay faithful to it."
)


def _fmt(turns: list[dict]) -> str:
    return "\n".join(
        f"[{t['start']:.0f}-{t['end']:.0f}] {t['speaker']}: {t['text'].strip()}" for t in turns
    )


def _speakers_in(turns: list[dict], start: float, end: float) -> set:
    return {t["speaker"] for t in turns if t["end"] > start and t["start"] < end}


def _ask(client, turns: list[dict], lo: int, hi: int, task: str, note: str = "") -> dict:
    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(task=task + note, lo=lo, hi=hi, transcript=_fmt(turns)),
        config={"response_mime_type": "application/json", "response_schema": SCHEMA},
    )
    clip = json.loads(resp.text)
    clip["start"], clip["end"] = float(clip["start"]), float(clip["end"])
    return clip


def _densest_exchange(turns: list[dict], lo: int, hi: int) -> dict | None:
    """a multi-speaker window with the most number of switches."""
    best = None
    for i in range(len(turns)):
        for j in range(i, len(turns)):
            start, end = turns[i]["start"], turns[j]["end"]
            if end - start > hi:
                break
            if end - start < lo:
                continue
            speakers = [turns[k]["speaker"] for k in range(i, j + 1)]
            if len(set(speakers)) < 2:
                continue
            switches = sum(a != b for a, b in zip(speakers, speakers[1:]))
            if best is None or switches > best[0]:
                best = (switches, start, end)
    if best:
        return {"start": float(best[1]), "end": float(best[2]), "title": ""}
    return None


def pick_clip(turns: list[dict], lo: int = 45, hi: int = 60, focus: str | None = None) -> dict:
    client = genai.Client()
    task = FOCUS_TASK.format(lo=lo, hi=hi, focus=focus) if focus else AUTO_TASK.format(lo=lo, hi=hi)

    clip = _ask(client, turns, lo, hi, task)
    if len(_speakers_in(turns, clip["start"], clip["end"])) >= 2:
        return clip

    note = (f'\n\nYour previous choice [{clip["start"]:.0f}-{clip["end"]:.0f}] had only one speaker. '
            "Pick a different window that clearly contains an exchange between at least two speakers.")
    retry = _ask(client, turns, lo, hi, task, note=note)
    if len(_speakers_in(turns, retry["start"], retry["end"])) >= 2:
        return retry

    fallback = _densest_exchange(turns, lo, hi)
    if fallback:
        fallback["title"] = clip["title"]
        return fallback
    return clip
