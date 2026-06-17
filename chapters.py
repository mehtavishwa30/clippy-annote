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
