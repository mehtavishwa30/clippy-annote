---
Title: Build a YouTube Shorts generator with Pyannote & FFmpeg
Description: Build a CLI tool that takes a video, automatically finds the most compelling clip, reframes it to 9:16 vertical, and burns in colour-coded karaoke-style captions, one colour per speaker.
---

## Table of contents
- Overview
- What we are building
- Quickstart
- Additional Resources
- Next Steps
- Conclusion

Author's note: [Skip overview if you wish to start building right away 💪]

## 💭 Overview

I know what you're thinking! "We have an abundance of captioning tools. Why build one?" Let's take a look at the problem statement before we dive into the development.

I love watching YouTube videos. With our attention spans shrinking in the AI and social media era, my instinct is to watch YouTube Shorts for quick reviews and highlights. Despite the video and audio quality being top-notch, the captioning is quite flimsy more often than not. The captions are a dump of every word that is being said. Overlaps are wonky, cross-talk becomes garbled reponses, and the identity of the speaker is unknown. The issue isn't with speech detection and STT solutions. The problem arises when the current industry standards fail to capture the essence of what is being said and by whom in podcasts and interviews.

**How do we fix it? 💡**

By adding a speaker intelligence layer underneath the transcription that serves as an insightful and accurate foundation for the raw data. For multi-speaker environments, 'who spoke when' is just as important as 'what was said'. This is where speaker diarization becomes our solution for our wonky captioning.

**What is speaker diarization? 🤔**

> Speaker diarization is the process of partitioning an audio recording of a conversation into distinct segments based on the identity of the speakers. Simply put, it answers the question, "who spoke when?" It transforms unstructured audio into organized, speaker-attributed transcripts (e.g., Speaker A, Speaker B).

**Why this matters 🪄**

A plain transcript answers "what was said." A diarized transcript answers "who said what, and when." That distinction unlocks:

| Without diarization | With diarization |
|---|---|
| Flat text search | Speaker-filtered semantic search |
| No attribution | Every answer cites the speaker |
| Opaque Q&A | Grounded, timestamped citations |

## 🚧 What we are building

This guide will show you how to build a a command-line tool that can:
- take a .mp4/wav video as input
- automatically finds the most compelling clip using AI
- generate captions based on diarized transcripts
- reframe the clip 9:16 vertical with blurred tabs
- and burns colour-coded karaoke-style captions, one colour per speaker into the video

You can view the full code for this guide [here](https://github.com/mehtavishwa30/clippy-annote).

### Tech stack

- Audio Pipeline: Pyannote (precision-2 for speaker diarization + OpenAI Whisper for STT)
- Video Rendering: FFmpeg
- Clip Window Intelligence: Gemini

### Architecture

```
[Input Video]    ➡️   [Extract Audio]   ➡️   [Pyannote: Who Spoke When]
                                                         ⬇️
[Render Color-Coded Video] ⬅️ [Map Timestamps] ⬅️ [Whisper: What Was Said]
```

## 📝 Quickstart

### Pre-requisites

- Python 3.11 or later
- FFmpeg installed and on your `PATH`
- A [pyannote.ai](https://pyannote.ai) account and API key
- A Gemini API (you can use your choice of LLM)
- Basic knowledge of Python

### Step 1: Setup a new project

a. Create a new project.

b. Navigate into your project directory.

c. Install the dependencies in `requirements.txt`.

```
pip install -r requirements.txt
```

### Step 2: Configure your API keys

Create a `.env` file and add your API keys as follows:
```
PYANNOTE_API_KEY=your_pyannote_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

We are using the precision-2 model for diarized transcripts which requires an API key from your Pyannote dashboard:
- Navogate to your Pyannote dashboard
- Go to API keys under `Manage` in left sidebar
- Enter name > Click `+ Create`

> **Never commit `.env` to version control.** Add it to `.gitignore` immediately:
> ```bash
> echo ".env" >> .gitignore
> ```

### Step 3: Write your `pyannote_client.py` file

Create a new file in your project directory called `pyannote_client.py` and edit the file such that the contents are as follows:

```py
"""upload audio, run precision-2 for diarization and transcription (whisper), poll for the result."""

import os
import time
import uuid

import requests

API = "https://api.pyannote.ai/v1"
# Transcription is only offered alongside the precision-2 diarization model.
# In community-1 model, you need to run transcription separately
DIARIZATION_MODEL = "precision-2"
TRANSCRIPTION_MODEL = "faster-whisper-large-v3-turbo"


class PyannoteError(RuntimeError):
    pass


def _headers() -> dict:
    key = os.environ.get("PYANNOTE_API_KEY")
    if not key:
        raise PyannoteError("api key is not set")
    return {"Authorization": f"Bearer {key}"}


def upload(path: str) -> str:
    """Upload a local audio file to pyannote.ai temporary storage (kept ~24h).

    Returns a media:// URL usable in a job request.
    """
    media_url = f"media://clippy-annote/{uuid.uuid4().hex}/{os.path.basename(path)}"
    resp = requests.post(f"{API}/media/input", json={"url": media_url}, headers=_headers())
    resp.raise_for_status()
    with open(path, "rb") as f:
        put = requests.put(resp.json()["url"], data=f)
    put.raise_for_status()
    return media_url


def submit(media_url: str) -> str:
    """Start a diarization and transcription job. Return the job ID."""
    body = {
        "url": media_url,
        "model": DIARIZATION_MODEL,
        "transcription": True,
        "transcriptionConfig": {"model": TRANSCRIPTION_MODEL},
    }
    resp = requests.post(f"{API}/diarize", json=body, headers=_headers())
    resp.raise_for_status()
    return resp.json()["jobId"]


def wait(job_id: str, poll_every: float = 5, timeout: float = 3600, on_poll=None) -> dict:
    """Poll until the job finishes, return its output:

        turnLevelTranscription: [{"speaker", "start", "end", "text"}, ...]
        wordLevelTranscription: [{"speaker", "start", "end", "text"}, ...]
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{API}/jobs/{job_id}", headers=_headers())
        resp.raise_for_status()
        job = resp.json()
        status = job["status"]
        if on_poll:
            on_poll(status)
        if status == "succeeded":
            return job["output"]
        if status in ("failed", "cancelled"):
            raise PyannoteError(f"job {job_id} {status}: {job.get('error', 'no details')}")
        time.sleep(poll_every)
    raise TimeoutError(f"job {job_id} did not finish within {timeout:.0f}s")
```

🧑‍🏫 **Understanding the key concepts in the code**

`pyannote_client.py` handles three things in sequence: Upload, Submit job, Poll

Upload the video to Pyannote platform

### Step 4: Extract the audio and render the shorts clip

Create a new file in your project directory called `render.py` and edit the file such that the contents are as follows:

```py
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
```

🧑‍🏫 **Understanding the key concepts in the code**

### Step 5: Write your main script `shorts.py`

Create a new file in your project directory called `shorts.py` and edit the file such that the contents are as follows:

```py
#!/usr/bin/env python3
"""clippy-annote generates Youtube shorts with colour-coded captions per speaker.
    using pyannote-ai and ffmpeg

    python shorts.py talk.mp4 -o short.mp4   # llm picks the highlight
    python shorts.py talk.mp4 --chapters   # list topic-based chapters
    python shorts.py talk.mp4 --topic "the Q&A on agent limits" -o out.mp4   # pass topic to generate shorts
    python shorts.py talk.mp4 --start 612 --end 668 -o out.mp4   # explicit window
"""

import argparse
import json
import os
import sys
import tempfile

from dotenv import load_dotenv

import captions
import chapters
import highlight
import pyannote_client as pa
import render


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:d}:{s:02d}"


def transcribe(path: str) -> dict:
    """diarize and transcribe `path`, caching the result next to the input"""
    cache = path + ".transcript.json"
    if os.path.exists(cache):
        log("using cached transcript")
        with open(cache) as f:
            return json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        audio = os.path.join(tmp, "audio.mp3")
        log("extracting audio")
        render.extract_audio(path, audio)
        log("uploading to pyannote")
        media = pa.upload(audio)
        job = pa.submit(media)
        log(f"diarizing and transcribing (job {job})")
        out = pa.wait(job, on_poll=lambda s: log(f"  {s}"))
    with open(cache, "w") as f:
        json.dump(out, f)
    return out


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Generate a YouTube Shorts with colour coded captions from a video recording.")
    ap.add_argument("input", help="source video")
    ap.add_argument("-o", "--output", default="short.mp4", help="output file (default: short.mp4)")
    ap.add_argument("--chapters", action="store_true", help="list topic-based chapters and exit")
    ap.add_argument("--topic", help="generate shorts based on this topic or question")
    ap.add_argument("--min", type=int, default=45, help="minimum clip length, seconds")
    ap.add_argument("--max", type=int, default=60, help="maximum clip length, seconds")
    ap.add_argument("--start", type=float, help="clip start in seconds; with --end, skips Gemini")
    ap.add_argument("--end", type=float, help="clip end in seconds; with --start, skips Gemini")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"no such file: {args.input}")
    if (args.start is None) != (args.end is None):
        sys.exit("pass --start and --end together, or neither")
    if args.start is not None and args.end <= args.start:
        sys.exit("--end must be after --start")
    if args.topic and args.start is not None:
        sys.exit("--topic can't be combined with --start/--end")

    out = transcribe(args.input)
    turns = out["turnLevelTranscription"]
    words = out["wordLevelTranscription"]
    if not turns:
        sys.exit("transcript was empty, nothing to do")

    if args.chapters:
        log("asking Gemini for chapters...") # added a progress log to avoid silent stall
        for c in chapters.make_chapters(turns):
            print(f'{_ts(c["start"])}  {c["title"]}')
        return

    if args.start is not None:
        start, end = args.start, args.end
        log(f"using given window  [{start:.0f}s-{end:.0f}s]")
    else:
        log(f"finding a clip about: {args.topic}..." if args.topic else "asking Gemini for the best highlight...")
        clip = highlight.pick_clip(turns, lo=args.min, hi=args.max, focus=args.topic)
        start, end = clip["start"], clip["end"]
        log(f'picked "{clip["title"]}"  [{start:.0f}s-{end:.0f}s]')

    clip_words = [
        {"start": w["start"] - start, "end": w["end"] - start,
         "text": w["text"], "speaker": w.get("speaker")}
        for w in words
        if w["start"] >= start and w["end"] <= end
    ]
    ass = os.path.splitext(args.output)[0] + ".ass"
    with open(ass, "w") as f:
        f.write(captions.build_ass(clip_words))

    log("rendering...")
    render.render(args.input, start, end - start, ass, args.output)
    os.remove(ass)
    log(f"done -> {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
```

🧑‍🏫 **Understanding the key concepts in the code**

### Step 6: Style your karaoke-style, colour-coded captions

Create a new file in your project directory called `captions.py` and edit the file such that the contents are as follows:

```py
"""This builds karaoke-syle word-level subtitles using .ass."""

WIDTH, HEIGHT = 1080, 1920
WHITE = "&H00FFFFFF"   # &HAABBGGRR colours

# colour-coded captions, distinct colour per speaker
SPEAKER_COLOURS = [
    "&H0000E5FF",  # amber
    "&H00FFE500",  # cyan
    "&H005AE63C",  # green
    "&H00C85AFF",  # pink
    "&H001E8CFF",  # orange
    "&H00FFAA50",  # sky
]

MAX_WORDS = 5  # words held on screen at once
MAX_GAP = 0.7  # seconds of silence that forces a new line

HEADER = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption, Helvetica, 76, {WHITE}, &H00000000, &H64000000, -1, 5, 0, 2, 120, 120, 560, 1

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, Effect, Text
"""


def _ts(t: float) -> str:
    cs = round(t * 100)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _colours(words: list[dict]) -> dict:
    order = []
    for w in words:
        if w.get("speaker") not in order:
            order.append(w.get("speaker"))
    return {spk: SPEAKER_COLOURS[i % len(SPEAKER_COLOURS)] for i, spk in enumerate(order)}


def _lines(words: list[dict]) -> list[list[dict]]:
    lines, cur = [], []
    for w in words:
        if cur and (len(cur) >= MAX_WORDS
                    or w.get("speaker") != cur[-1].get("speaker")
                    or w["start"] - cur[-1]["end"] > MAX_GAP
                    or cur[-1]["text"].strip().endswith((".", "?", "!"))):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _event(layer: int, start: float, end: float, text: str) -> str:
    return f"Dialogue: {layer},{_ts(start)},{_ts(end)},Caption,,,,{text}"


def build_ass(words: list[dict]) -> str:
    """words: clip-relative [{start, end, text, speaker}, ...]. Returns the full .ass document."""
    colour = _colours(words)
    events = []
    for line in _lines(words):
        spk = colour[line[0].get("speaker")]
        texts = [w["text"].strip() for w in line]
        # caption in speaker's colour
        events.append(_event(0, line[0]["start"], line[-1]["end"], f"{{\\c{spk}}}" + " ".join(texts)))
        # one event per word, white while spoken
        for i, w in enumerate(line):
            lit = list(texts)
            lit[i] = f"{{\\c{WHITE}}}{texts[i]}{{\\c{spk}}}"
            events.append(_event(1, w["start"], w["end"], f"{{\\c{spk}}}" + " ".join(lit)))
    return HEADER + "\n".join(events) + "\n"
```

🧑‍🏫 **Understanding the key concepts in the code**

### Step 7: Write the logic to list chapters based on topics in the video

Create a new file in your project directory called `chapters.py` and edit the file such that the contents are as follows:

```py
"""Gemini breaks down transcript into chapters"""

import json

from google import genai

MODEL = "gemini-2.5-flash"

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start": {"type": "NUMBER", "description": "Chapter start in seconds."},
            "end": {"type": "NUMBER", "description": "Chapter end in seconds."},
            "title": {"type": "STRING",
                      "description": "Short topic. For a Q&A turn, the question itself, condensed."},
        },
        "required": ["start", "end", "title"],
    },
}

PROMPT = """\
Here is a speaker-labelled transcript of a video, one turn per line as [start-end] SPEAKER: text.

- Split it into chapters from the conversation. Each chapter is one topic or a question and answer 
segment in case of in an interview/Q&A.
- Title those with the topic/question. Cover the whole transcript in order of timestamps, with no gaps or overlaps.
- Keep titles short and specific.
- Produce between 8 and 12 chapters total. group related turns rather than giving each its own chapter.

{transcript}"""


def _fmt(turns: list[dict]) -> str:
    return "\n".join(
        f"[{t['start']:.0f}-{t['end']:.0f}] {t['speaker']}: {t['text'].strip()}" for t in turns
    )


def make_chapters(turns: list[dict]) -> list[dict]:
    client = genai.Client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(transcript=_fmt(turns)),
        config={"response_mime_type": "application/json", "response_schema": SCHEMA},
    )
    return json.loads(resp.text)
```

🧑‍🏫 **Understanding the key concepts in the code**

### Step 8: Write the Gemini clip-picker engine

Create a new file in your project directory called `highlight.py` and edit the file such that the contents are as follows:

```py
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
```

🧑‍🏫 **Understanding the key concepts in the code**

### Testing your CLI tool

a. Pick a sample video and save it at the same root directory as the project files.

b. Run the project using one of the following options provided by the CLI:

* Use the default auto-mode to let Gemini pick the best window for the clip.

  ```sh
  python shorts.py sample.mp4 -o short.mp4   # Gemini picks the clip
  ```

* Use the manual mode to pass start and end times for the clip window.

  ```sh
  python shorts.py sample.mp4 --start 34 --end 93  -o short.mp4  # explicitly pick a viral window
  ```
  
* List chapter markers. LLM analyzes the video and shows various topics/questions covered.

  ```sh
  python shorts.py sample.mp4 --chapters
  ```
  Your output should look something like:

  ```sh
  using cached transcript
  asking Gemini for chapters
  0:00  What are your agents doing today?
  1:28  Tell us about Base 10 and how you got there.
  4:25  How much responsibility can agents handle?
  5:55  How do you set up your agent harness?
  9:25  Using different models and inter-agent communication
  17:52  What can agents do in ML research, and what is the human role?
  24:26  The future of context windows and KV cache compaction
  27:44  Where do you see AI going in the next few years?
  37:32  The importance of UI/UX, automation, and agent autonomy
  41:37  Conclusion
  ```

* Pick a window based on a specific chapter.

  ```sh
  python shorts.py sample.mp4 --topic "How do you set up your agent harness?" -o short.mp4
  ```

## Additional Resources

- [Pyannote Docs](https://docs.pyannote.ai/introduction)
- [FFmpeg docs](https://www.ffmpeg.org/documentation.html)
- [Pyannote speaker diarization toolkit](https://github.com/pyannote/pyannote-audio)

## Next Steps

Awesome. Now that you've learned how to build a CLI tool to generate YouTube shorts using Pyannote and FFmpeg, you can:
- Turn it into a Streamlit/Gradio app for a cleaner UI/UX interface for non-technical users
- Add more tools on top of the speaker intelligence such as: RAG to ask questions, AI note-taker that shares/stores a summary of the insights in Notion, etc.

## Conclusion

Speaker diarization is just one piece in the speaker intelligence puzzel and we're excited to see what is possible!
