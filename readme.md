# clippy-annote

clippy-annote is a CLI tool written in Python that allows you create beautiful YouTube Shorts with colour-coded captions per speaker.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in PYANNOTE_API_KEY and GEMINI_API_KEY
```

## Try it out

clippy-annote CLI provides various options to steer the generation of your YouTube Shorts.

### Options

1. Use the default auto-mode to let Gemini pick the best window for the clip.

  ```sh
  python shorts.py sample.mp4 -o short.mp4   # Gemini picks the clip
  ```

2. Use the manual mode to pass start and end times for the clip window.

  ```
  python shorts.py sample.mp4 --start 34 --end 93  -o short.mp4  # explicitly pick a viral window
  ```
  
3. List chapter markers. LLM analyzes the video and shows various topics/questions covered.

  ```sh
  python shorts.py sample.mp4 --chapters
  ```

4. Pick a window based on a specific chapter.

  ```sh
  python shorts.py sample.mp4 --topic "How do agents interact securely across users?" -o short.mp4
  ```

## How it works

| Step | File |
| --- | --- |
| 16 kHz mono audio for upload | `render.py` |
| `precision-2` diarization + `faster-whisper-large-v3-turbo` transcription | `pyannote_client.py` |
| Topical / Q&A chapters | `chapters.py` |
| Pick the clip window (auto, `--topic`, guaranteed multi-speaker) | `highlight.py` |
| Word-level karaoke `.ass`, one colour per speaker | `captions.py` |
| Cut → blurred 9:16 reframe → burn captions | `render.py` |

The clip picker always returns a multi-speaker exchange: if Gemini's pick is
single-speaker it is re-asked once, then falls back to the densest exchange in
the transcript. pyannote.ai only offers transcription with `precision-2`, so
that one job covers both diarization and the word timings the captions need.
