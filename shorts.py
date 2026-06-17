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
