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
