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
