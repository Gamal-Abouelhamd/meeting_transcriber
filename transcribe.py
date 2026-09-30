#!/usr/bin/env python3
r"""
transcribe.py
----------------------
Transcribes a long meeting recording (Egyptian Arabic <-> English
code-switching) using Gemini, Speechmatics, or QwenCleo-ASR locally.

SETUP
  1. python -m pip install -r requirements.txt
  2. Install ffmpeg and make sure the `ffmpeg` command is on your PATH if
     you use --chunk-minutes or the local QwenCleo provider.
  3. Get a Gemini key from https://aistudio.google.com/apikey and/or a
     Speechmatics key from https://portal.speechmatics.com/
  4. Put the key(s) in the .env file next to this script:
       GEMINI_API_KEY=your_key_here
       SPEECHMATICS_API_KEY=your_key_here
     An existing environment variable takes precedence over .env.

USAGE
  python transcribe.py "meeting.mp4" --chunk-minutes 20
  python transcribe.py meeting.mp3 -o transcript.txt
  python transcribe.py meeting.mp3 --gemini-3-flash-preview
  python transcribe.py meeting.mp3 --provider speechmatics
  .\.venv-qwencleo\Scripts\python.exe transcribe.py meeting.mp3 --provider qwencleo

NOTES
  - Long recordings can exceed the model's output limit even when the
    input fits. Use --chunk-minutes 20 for this meeting (requires ffmpeg).
  - Chunk timestamps and numbered speaker labels are local to each chunk;
    the chunk heading gives its approximate start in the full recording.
  - Completed chunks are saved immediately. An interrupted run leaves a
    partial transcript; restarting does not automatically resume it.
  - The Gemini prompt below explicitly tells the model to KEEP English words as
     written English instead of translating or transliterating them into
     Arabic script, which is the main failure mode with code-switched audio.
  - Speechmatics uses its Melia multilingual model with Arabic and English
    language hints and speaker diarization by default.
  - QwenCleo runs entirely on this computer. It automatically converts the
    recording to 16 kHz mono WAV and uses short overlapping windows suitable
    for an 8 GB GPU. It provides approximate window timestamps, not speakers.
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from speechmatics.batch import (
        AsyncClient as SpeechmaticsClient,
        Model as SpeechmaticsModel,
        Transcript as SpeechmaticsTranscript,
        TranscriptionConfig as SpeechmaticsTranscriptionConfig,
    )
except ImportError:
    SpeechmaticsClient = None
    SpeechmaticsModel = None
    SpeechmaticsTranscript = None
    SpeechmaticsTranscriptionConfig = None

try:
    from qwencleo_asr import QwenCleoASR
except ImportError:
    QwenCleoASR = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


GEMINI_DEFAULT_MODEL = "gemini-3-flash-preview"
SPEECHMATICS_DEFAULT_MODEL = "melia-1"
SPEECHMATICS_MODELS = ("melia-1", "enhanced", "standard")
QWENCLEO_DEFAULT_MODEL = "mohammedaly22/QwenCleo-ASR"
QWENCLEO_DEFAULT_CHUNK_SECONDS = 20.0
QWENCLEO_DEFAULT_OVERLAP_SECONDS = 2.0
INCOMPLETE_TRANSCRIPT_MARKER = "INCOMPLETE TRANSCRIPT - transcription in progress."


PROMPT = """You are transcribing an audio recording of a business meeting.
The speakers are Egyptian, and they naturally code-switch between Egyptian
Arabic and English within the same sentences.

Rules:
1. Write Egyptian Arabic speech in Arabic script (not Franco-Arabic).
2. Write English words, phrases, or sentences exactly as spoken, in Latin
   script — do NOT translate them into Arabic and do NOT transliterate
   them into Arabic letters. Keep the mixed-language sentences mixed.
3. Do not translate anything into a single language. Preserve the
   original code-switching exactly as spoken.
4. If you can distinguish different speakers, label them Speaker 1,
   Speaker 2, etc. (or use names if they are stated in the audio).
5. Insert an approximate timestamp like [00:12:30] every time the topic
   or speaker changes, or roughly every 1-2 minutes.
6. Skip filler sounds (uh, um) but keep meaningful disfluencies if they
   affect meaning.
7. Output plain text only — no commentary, no summary, just the
   transcript.
8. Transcribe the entire clip through its end. Mark unintelligible speech
   as [inaudible]; never invent words. Timestamps start at zero for this clip.
"""


def delete_upload(client, name):
    """Best-effort cleanup without masking an earlier error."""
    try:
        client.files.delete(name=name)
    except Exception:
        print(f"Warning: could not delete uploaded file {name}.", file=sys.stderr)


def upload_and_wait(client: "genai.Client", path: Path, timeout: int = 600):
    """Upload a file via the File API and wait until it's ready to use."""
    print(f"  Uploading {path.name} ...")
    uploaded = client.files.upload(file=str(path))

    name = uploaded.name
    deadline = time.monotonic() + timeout
    try:
        while uploaded.state and uploaded.state.name == "PROCESSING":
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out processing {path.name}")
            time.sleep(3)
            uploaded = client.files.get(name=name)

        if not uploaded.state or uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini failed to process {path.name}: {uploaded.state}")
    except BaseException:
        delete_upload(client, name)
        raise

    return uploaded


def transcribe_gemini_file(client: "genai.Client", model: str, path: Path) -> str:
    audio_file = upload_and_wait(client, path)
    print(f"  Transcribing {path.name} with {model} ...")

    try:
        response = client.models.generate_content(
            model=model,
            contents=[PROMPT, audio_file],
            config=types.GenerateContentConfig(
                # Keep the model's default temperature, as recommended for Gemini 3.
                max_output_tokens=65536,
            ),
        )
        candidate = response.candidates[0] if response.candidates else None
        reason = candidate.finish_reason if candidate else None
        reason_name = getattr(reason, "value", reason)
        if reason_name == "MAX_TOKENS":
            raise RuntimeError("Transcript hit the output limit. Retry with smaller --chunk-minutes.")
        if reason_name != "STOP":
            raise RuntimeError(f"Transcription did not finish normally: {reason_name}")
        text = response.text
        if not text or not text.strip():
            raise RuntimeError(f"Gemini returned an empty transcript for {path.name}")
        return text
    finally:
        delete_upload(client, audio_file.name)


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _join_speechmatics_items(items: list[tuple[object, object]]) -> str:
    """Join Speechmatics word/punctuation results into readable text."""
    text = ""
    opening_punctuation = ("(", "[", "{", "«", "“")

    for result, alternative in items:
        content = getattr(alternative, "content", "")
        if not content:
            continue

        result_type = getattr(result, "type", "")
        attaches_to = getattr(result, "attaches_to", None)
        if result_type == "punctuation" or attaches_to == "previous":
            if attaches_to == "next":
                if text and not text.endswith((" ", "\n")):
                    text += " "
                text += content
            else:
                text = text.rstrip() + content
        else:
            if text and not text.endswith((" ", "\n", *opening_punctuation)):
                text += " "
            text += content

    return text.strip()


def format_speechmatics_transcript(transcript: object, segment_seconds: int = 90) -> str:
    """Add timestamps and stable speaker labels to a Speechmatics transcript."""
    segments: list[tuple[float, str | None, list[tuple[object, object]]]] = []
    current_items: list[tuple[object, object]] = []
    current_speaker: str | None = None
    segment_start = 0.0

    for result in getattr(transcript, "results", []):
        alternatives = getattr(result, "alternatives", None)
        if not alternatives:
            continue
        alternative = alternatives[0]
        if not getattr(alternative, "content", ""):
            continue

        speaker = getattr(alternative, "speaker", None)
        start_time = float(getattr(result, "start_time", 0.0))
        is_word = getattr(result, "type", "") != "punctuation"
        should_split = bool(current_items) and (
            speaker != current_speaker
            or (is_word and start_time - segment_start >= segment_seconds)
        )
        if should_split:
            segments.append((segment_start, current_speaker, current_items))
            current_items = []

        if not current_items:
            segment_start = start_time
            current_speaker = speaker
        current_items.append((result, alternative))

    if current_items:
        segments.append((segment_start, current_speaker, current_items))

    speaker_numbers: dict[str, int] = {}
    lines = []
    for start_time, speaker, items in segments:
        text = _join_speechmatics_items(items)
        if not text:
            continue
        prefix = f"[{format_timestamp(start_time)}]"
        if speaker:
            if speaker not in speaker_numbers:
                speaker_numbers[speaker] = len(speaker_numbers) + 1
            prefix += f" Speaker {speaker_numbers[speaker]}:"
        lines.append(f"{prefix} {text}")

    return "\n".join(lines)


async def delete_speechmatics_job(client: object, job_id: str) -> None:
    """Best-effort cleanup without masking an earlier error."""
    try:
        await client.delete_job(job_id)
    except Exception:
        print(f"Warning: could not delete Speechmatics job {job_id}.", file=sys.stderr)


async def transcribe_speechmatics_file(
    client: object,
    model: str,
    language: str,
    language_hints: list[str],
    path: Path,
) -> str:
    config_kwargs = {
        "language": language,
        "model": SpeechmaticsModel(model),
        "diarization": "speaker",
    }
    if model == "melia-1" and language_hints:
        config_kwargs.update(
            language_hints=language_hints,
            language_hints_strict=False,
        )
    config = SpeechmaticsTranscriptionConfig(**config_kwargs)

    print(f"  Uploading {path.name} to Speechmatics ...")
    job = await client.submit_job(str(path), transcription_config=config)
    print(f"  Transcribing {path.name} with Speechmatics {model} (job {job.id}) ...")
    try:
        transcript = await client.wait_for_completion(job.id, timeout=3600)
        if not isinstance(transcript, SpeechmaticsTranscript):
            raise RuntimeError("Speechmatics returned an unexpected transcript format")
        text = format_speechmatics_transcript(transcript)
        if not text.strip():
            raise RuntimeError(f"Speechmatics returned an empty transcript for {path.name}")
        return text
    finally:
        await delete_speechmatics_job(client, job.id)


def resolve_qwencleo_runtime(device: str, dtype: str) -> tuple[str, str]:
    """Resolve safe QwenCleo device/dtype defaults for the current machine."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing. Run .\\install_qwencleo.ps1 first.") from exc

    if device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        resolved_device = "cuda:0"
    else:
        resolved_device = device

    if resolved_device != "cpu" and not resolved_device.startswith("cuda:"):
        raise RuntimeError("QwenCleo device must be auto, cpu, cuda, or cuda:N.")

    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see the NVIDIA GPU. "
            "Run .\\install_qwencleo.ps1 and check its CUDA verification output."
        )
    if resolved_device.startswith("cuda:"):
        try:
            device_index = int(resolved_device.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError("QwenCleo CUDA device must look like cuda:0.") from exc
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device_index} is unavailable; found {torch.cuda.device_count()} GPU(s)."
            )

    if dtype == "auto":
        if resolved_device.startswith("cuda"):
            supports_bfloat16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            resolved_dtype = "bfloat16" if supports_bfloat16 else "float16"
        else:
            resolved_dtype = "float32"
    else:
        resolved_dtype = dtype

    if resolved_device == "cpu" and resolved_dtype == "float16":
        raise RuntimeError("float16 is not suitable for CPU inference; use --local-dtype float32.")

    return resolved_device, resolved_dtype


def prepare_qwencleo_audio(path: Path, workdir: Path) -> Path:
    """Extract audio as 16 kHz mono PCM for local transcription."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required for local QwenCleo transcription; install it and add it to PATH."
        )

    wav_path = workdir / "recording_16khz_mono.wav"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while preparing audio:\n{result.stderr}")
    return wav_path


def split_wav_overlapping(
    wav_path: Path,
    chunk_seconds: float,
    overlap_seconds: float,
    workdir: Path,
) -> list[tuple[Path, float, float]]:
    """Split a PCM WAV and return (path, start seconds, end seconds) triples."""
    if chunk_seconds <= 0:
        raise ValueError("local chunk seconds must be greater than zero")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("local overlap seconds must be >= 0 and smaller than chunk seconds")

    chunks: list[tuple[Path, float, float]] = []
    with wave.open(str(wav_path), "rb") as source:
        sample_rate = source.getframerate()
        total_frames = source.getnframes()
        frames_per_chunk = max(1, round(chunk_seconds * sample_rate))
        frames_per_hop = max(1, round((chunk_seconds - overlap_seconds) * sample_rate))
        params = source.getparams()

        start_frame = 0
        index = 0
        while start_frame < total_frames:
            remaining_frames = total_frames - start_frame
            if index > 0 and remaining_frames < round(0.3 * sample_rate):
                break

            source.setpos(start_frame)
            frames = source.readframes(min(frames_per_chunk, remaining_frames))
            chunk_path = workdir / f"qwencleo_chunk_{index:05d}.wav"
            with wave.open(str(chunk_path), "wb") as chunk:
                chunk.setparams(params)
                chunk.writeframes(frames)

            frame_count = len(frames) // (params.sampwidth * params.nchannels)
            chunks.append(
                (chunk_path, start_frame / sample_rate, (start_frame + frame_count) / sample_rate)
            )
            start_frame += frames_per_hop
            index += 1

    if not chunks:
        raise RuntimeError("The recording contains no audio to transcribe")
    return chunks


def deduplicate_overlap(previous: str, current: str, max_words: int = 40) -> str:
    """Remove an exact repeated phrase caused by overlapping audio windows."""
    previous_words = previous.split()
    current_words = current.split()
    punctuation = ".,!?;:\"'()[]{}<>،؛؟…«»“”"

    def normalized(words: list[str]) -> list[str]:
        return [word.strip(punctuation).casefold() for word in words]

    previous_normalized = normalized(previous_words)
    current_normalized = normalized(current_words)
    limit = min(max_words, len(previous_words), len(current_words))
    for count in range(limit, 1, -1):
        if previous_normalized[-count:] == current_normalized[:count]:
            return " ".join(current_words[count:]).strip()
    return current.strip()


def decode_utf8_preserving_valid_text(data: bytes) -> tuple[str, list[int]]:
    """Decode UTF-8 while dropping only malformed byte sequences."""
    parts: list[str] = []
    invalid_offsets: list[int] = []
    cursor = 0
    while cursor < len(data):
        try:
            parts.append(data[cursor:].decode("utf-8"))
            break
        except UnicodeDecodeError as exc:
            valid_end = cursor + exc.start
            invalid_end = cursor + exc.end
            parts.append(data[cursor:valid_end].decode("utf-8"))
            invalid_offsets.extend(range(valid_end, invalid_end))
            cursor = invalid_end
    return "".join(parts), invalid_offsets


def finalize_transcript_file(output_path: Path, *, trim_leading: bool = False) -> None:
    """Remove the progress marker and atomically normalize the transcript to UTF-8."""
    data = output_path.read_bytes()
    marker = INCOMPLETE_TRANSCRIPT_MARKER.encode("utf-8")
    if not data.startswith(marker):
        raise RuntimeError(f"Progress marker is missing from {output_path}")

    first_line_end = data.find(b"\n")
    if first_line_end < 0:
        raise RuntimeError(f"Transcript contains only a progress marker: {output_path}")

    text, invalid_offsets = decode_utf8_preserving_valid_text(data[first_line_end + 1:])
    if invalid_offsets:
        offsets = ", ".join(str(first_line_end + 1 + offset) for offset in invalid_offsets)
        print(
            f"Warning: removed {len(invalid_offsets)} malformed UTF-8 byte(s) "
            f"from {output_path} at offset(s): {offsets}.",
            file=sys.stderr,
        )
    if trim_leading:
        text = text.lstrip()

    # Write a complete replacement before swapping it into place. If this
    # process is interrupted, the original progress file remains recoverable.
    temporary_path = output_path.with_name(f".{output_path.name}.finalizing")
    try:
        temporary_path.write_bytes(text.encode("utf-8"))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def transcribe_qwencleo_recording(model: object, args, output_path: Path) -> None:
    """Convert, window, and locally transcribe a complete recording."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("Preparing 16 kHz mono audio for QwenCleo ...")
        wav_path = prepare_qwencleo_audio(args.audio, tmp_path)
        chunks = split_wav_overlapping(
            wav_path,
            args.local_chunk_seconds,
            args.local_overlap_seconds,
            tmp_path,
        )
        print(f"  {len(chunks)} local window(s) created.")

        previous_text = ""
        with output_path.open("x", encoding="utf-8") as output:
            output.write(f"{INCOMPLETE_TRANSCRIPT_MARKER}\n")
            output.flush()
            for index, (chunk_path, start, end) in enumerate(chunks, start=1):
                print(
                    f"[{index}/{len(chunks)}] Transcribing "
                    f"{format_timestamp(start)}-{format_timestamp(end)}"
                )
                result = model.transcribe(
                    str(chunk_path),
                    language=args.language,
                    normalize=args.local_normalize,
                )
                raw_text = getattr(result, "text", "").strip()
                if not raw_text:
                    raise RuntimeError(f"QwenCleo returned an empty transcript for window {index}")

                text = deduplicate_overlap(previous_text, raw_text) if previous_text else raw_text
                if text:
                    output.write(f"\n[{format_timestamp(start)}] {text}")
                    output.flush()
                previous_text = raw_text

        # Remove the in-progress marker only after every window succeeded.
        finalize_transcript_file(output_path, trim_leading=True)


def split_audio(path: Path, chunk_minutes: int, workdir: Path) -> list[Path]:
    """Extract the first audio stream and split it into MP3 chunks."""
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for --chunk-minutes; install it and add it to PATH.")
    pattern = str(workdir / "chunk_%05d.mp3")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(path),
        "-map", "0:a:0", "-vn", "-c:a", "libmp3lame", "-b:a", "64k",
        "-f", "segment", "-segment_time", str(chunk_minutes * 60),
        "-reset_timestamps", "1", pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    chunks = sorted(workdir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("ffmpeg produced no audio chunks")
    return chunks


def parse_language_hints(value: str) -> list[str]:
    """Parse a comma-separated language list while preserving order."""
    return list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to the audio or video recording")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Output text file (default: <audio_name>.transcript.txt)")
    parser.add_argument(
        "--provider", choices=("gemini", "speechmatics", "qwencleo"), default="gemini",
        help="Transcription provider to use (default: gemini)",
    )
    parser.add_argument("--model", default=None,
                         help="Provider model/checkpoint (uses a provider-specific default)")
    parser.add_argument("--language", default=None,
                         help="Speechmatics code, or QwenCleo language name/auto")
    parser.add_argument("--language-hints", default="ar,en",
                         help="Comma-separated Melia language hints (default: ar,en; empty disables)")
    parser.add_argument("--chunk-minutes", type=int, default=None,
                         help="Split audio into chunks of N minutes before "
                              "transcribing (requires ffmpeg). Improves "
                              "accuracy on very long recordings.")
    parser.add_argument(
        "--local-device", default="auto",
        help="QwenCleo device: auto, cuda, cuda:N, or cpu (default: auto)",
    )
    parser.add_argument(
        "--local-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto",
        help="QwenCleo numeric type (default: auto)",
    )
    parser.add_argument(
        "--local-chunk-seconds", type=float, default=QWENCLEO_DEFAULT_CHUNK_SECONDS,
        help="QwenCleo window size (default: 20)",
    )
    parser.add_argument(
        "--local-overlap-seconds", type=float, default=QWENCLEO_DEFAULT_OVERLAP_SECONDS,
        help="QwenCleo window overlap (default: 2)",
    )
    parser.add_argument(
        "--local-max-new-tokens", type=int, default=256,
        help="QwenCleo token limit per window (default: 256)",
    )
    parser.add_argument(
        "--local-normalize", action="store_true",
        help="Apply QwenCleo's Egyptian-aware text normalization",
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        sys.exit(f"File not found: {args.audio}")
    if args.chunk_minutes is not None and args.chunk_minutes <= 0:
        parser.error("--chunk-minutes must be greater than zero")
    if args.provider == "gemini" and (genai is None or types is None):
        sys.exit("Missing Gemini dependency. Run: python -m pip install -r requirements.txt")
    if args.provider == "speechmatics" and (
        SpeechmaticsClient is None
        or SpeechmaticsModel is None
        or SpeechmaticsTranscript is None
        or SpeechmaticsTranscriptionConfig is None
    ):
        sys.exit("Missing Speechmatics dependency. Run: python -m pip install -r requirements.txt")
    if args.provider == "qwencleo" and QwenCleoASR is None:
        sys.exit(
            "Missing QwenCleo dependencies. Run .\\install_qwencleo.ps1, then use "
            ".\\.venv-qwencleo\\Scripts\\python.exe to run this script."
        )
    if args.provider != "qwencleo" and load_dotenv is None:
        sys.exit("Missing dependency. Run: python -m pip install -r requirements.txt")

    if args.provider == "gemini":
        args.model = args.model or GEMINI_DEFAULT_MODEL
    elif args.provider == "speechmatics":
        args.model = args.model or SPEECHMATICS_DEFAULT_MODEL
        if args.model not in SPEECHMATICS_MODELS:
            parser.error(
                "Speechmatics --model must be one of: " + ", ".join(SPEECHMATICS_MODELS)
            )
        args.language = args.language or ("multi" if args.model == "melia-1" else "ar")
        if args.model == "melia-1" and args.language != "multi":
            parser.error("Speechmatics melia-1 requires --language multi")
        if args.model != "melia-1" and args.language == "multi":
            parser.error("Speechmatics --language multi requires --model melia-1")
        args.language_hints = parse_language_hints(args.language_hints)
    else:
        args.model = args.model or QWENCLEO_DEFAULT_MODEL
        if args.chunk_minutes is not None:
            parser.error(
                "QwenCleo automatically uses short windows; configure them with "
                "--local-chunk-seconds instead of --chunk-minutes"
            )
        if args.local_chunk_seconds <= 0:
            parser.error("--local-chunk-seconds must be greater than zero")
        if not 0 <= args.local_overlap_seconds < args.local_chunk_seconds:
            parser.error(
                "--local-overlap-seconds must be >= 0 and smaller than --local-chunk-seconds"
            )
        if args.local_max_new_tokens <= 0:
            parser.error("--local-max-new-tokens must be greater than zero")
        if args.language is None:
            args.language = "Arabic"
        elif args.language.casefold() in ("auto", "none"):
            args.language = None

    output_path = args.output or args.audio.with_suffix(".transcript.txt")
    if output_path.resolve() == args.audio.resolve():
        parser.error("Output must not overwrite the input recording")
    if output_path.exists():
        parser.error(f"Output already exists: {output_path}. Choose a new filename with -o.")

    api_key = None
    if args.provider != "qwencleo":
        load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
        key_name = "GEMINI_API_KEY" if args.provider == "gemini" else "SPEECHMATICS_API_KEY"
        api_key = os.environ.get(key_name, "").strip()
        if not api_key:
            sys.exit(f"Set {key_name} in the .env file next to transcribe.py, or in your environment.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_transcription(args, output_path, api_key))

    print(f"\nDone. Transcript saved to: {output_path}")


async def run_transcription(args, output_path: Path, api_key: str | None) -> None:
    if args.provider == "gemini":
        with genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=600_000)) as client:
            async def transcribe_one(path: Path) -> str:
                return transcribe_gemini_file(client, args.model, path)

            await transcribe_recording(args, output_path, transcribe_one)
    elif args.provider == "speechmatics":
        async with SpeechmaticsClient(api_key=api_key) as client:
            async def transcribe_one(path: Path) -> str:
                return await transcribe_speechmatics_file(
                    client,
                    args.model,
                    args.language,
                    args.language_hints,
                    path,
                )

            await transcribe_recording(args, output_path, transcribe_one)
    else:
        device, dtype = resolve_qwencleo_runtime(args.local_device, args.local_dtype)
        print(f"Loading {args.model} locally on {device} ({dtype}) ...")
        print("  The first run downloads the model weights and can take several minutes.")
        model = QwenCleoASR(
            model_id=args.model,
            device=device,
            dtype=dtype,
            max_new_tokens=args.local_max_new_tokens,
            default_language=args.language,
        )
        transcribe_qwencleo_recording(model, args, output_path)


async def transcribe_recording(args, output_path: Path, transcribe_one) -> None:
    if args.chunk_minutes:
        with tempfile.TemporaryDirectory() as tmp:
            await transcribe_chunks(transcribe_one, args, Path(tmp), output_path)
    else:
        text = await transcribe_one(args.audio)
        with output_path.open("x", encoding="utf-8") as output:
            output.write(text)


async def transcribe_chunks(transcribe_one, args, tmp_path: Path, output_path: Path) -> None:
    print(f"Splitting into {args.chunk_minutes}-minute chunks ...")
    chunks = split_audio(args.audio, args.chunk_minutes, tmp_path)
    print(f"  {len(chunks)} chunk(s) created.")
    with output_path.open("x", encoding="utf-8") as output:
        output.write(f"{INCOMPLETE_TRANSCRIPT_MARKER}\n")
        output.flush()
        for i, chunk in enumerate(chunks, start=1):
            print(f"[{i}/{len(chunks)}] Processing {chunk.name}")
            text = await transcribe_one(chunk)
            offset_min = (i - 1) * args.chunk_minutes
            output.write(f"\n\n=== Chunk {i} (starts ~{offset_min} min; local timestamps/speakers) ===\n")
            output.write(text)
            output.flush()
    # Remove the in-progress marker only after every chunk succeeded.
    finalize_transcript_file(output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled. Any completed chunks remain in the output file.")
    except Exception as exc:
        sys.exit(f"Error: {exc}")
