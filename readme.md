# Meeting Transcription with Gemini, Speechmatics, or QwenCleo

A Python script that transcribes meeting recordings while preserving Egyptian Arabic and English as spoken. Gemini remains the default provider; Speechmatics and fully local QwenCleo-ASR are available through the same command-line interface.

## 1. Install dependencies

Use Python 3.10 or newer. Open a terminal in the project folder and run:

```powershell
python -m pip install -r requirements.txt
```

Splitting recordings with `--chunk-minutes` requires **ffmpeg** on your `PATH`. It is a separate program, not a Python dependency. Check that it is installed:

```powershell
ffmpeg -version
```

If it is missing, download it from the [ffmpeg website](https://ffmpeg.org/download.html), add its `bin` folder to `PATH`, and open a new terminal.

## 2. Set up your API key once

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey), the [Speechmatics Portal](https://portal.speechmatics.com/), or both. Open `.env` next to `transcribe.py` and paste each key you plan to use after `=`:

```dotenv
GEMINI_API_KEY=your_api_key_here
SPEECHMATICS_API_KEY=your_api_key_here
```

Save the file. The script loads it automatically using [python-dotenv](https://bbc2.github.io/python-dotenv/), even when launched from another folder. You do not need to enter the key in the terminal each time.

If a key is already set in your terminal environment, it takes precedence over `.env`. Open a new terminal if you previously set an outdated temporary value.

The `.gitignore` file excludes `.env`, local environment variants, recordings, and `.txt` output files. The `.env.example` template contains no key and can be shared in Git.

## 3. Transcribe a meeting

### Gemini (default)

Replace the filename with your recording's name, or provide its full path in quotes:

```powershell
python transcribe.py "meeting.mp4" --chunk-minutes 20
```

This splits the audio into chunks of approximately 20 minutes and saves the transcript next to the recording as `meeting.transcript.txt`.

To choose the output filename or location:

```powershell
python transcribe.py "D:\Meetings\weekly meeting.mp3" --chunk-minutes 20 -o "weekly.transcript.txt"
```

### Speechmatics

Choose Speechmatics with `--provider speechmatics`:

```powershell
python transcribe.py "meeting.mp4" --provider speechmatics
```

The Speechmatics default is the `melia-1` multilingual model with Arabic and English hints plus speaker diarization. It is designed for code-switched audio, so splitting is optional. To narrow or change the expected languages:

```powershell
python transcribe.py "meeting.mp4" --provider speechmatics --language-hints ar,en
```

To use a single-language Speechmatics model instead:

```powershell
python transcribe.py "meeting.mp4" --provider speechmatics --model enhanced --language ar
```

### QwenCleo-ASR (local, no API key)

QwenCleo uses a separate environment so its CUDA/PyTorch packages do not disturb the cloud-provider installation. On Windows PowerShell, run the setup once:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_qwencleo.ps1
```

To download the model during setup instead of waiting for the first transcription, add `-DownloadModel`:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_qwencleo.ps1 -DownloadModel
```

The model weights are about 4 GB. If the connection is interrupted, run the same command again; the installer resumes the cached partial download. On especially slow links, increase the per-read timeout (in seconds):

```powershell
powershell -ExecutionPolicy Bypass -File .\install_qwencleo.ps1 -DownloadModel -DownloadTimeoutSeconds 300
```

Then transcribe locally:

```powershell
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo
```

.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "3.mp4" `
  --provider qwencleo `
  --local-chunk-seconds 60 `
  --local-overlap-seconds 5 `
  --local-max-new-tokens 768 `
  -o "3.60s.transcript.txt"

The local provider defaults are tuned for this machine's NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM: CUDA, BF16, 20-second windows, and 2 seconds of overlap. QwenCleo automatically keeps the model loaded while processing all windows. No recording is uploaded and no API key is needed.

Useful overrides:

```powershell
# Automatic language detection instead of the recommended Arabic hint
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo --language auto

# Smaller windows if CUDA runs out of memory
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo --local-chunk-seconds 15 --local-overlap-seconds 2

# CPU fallback (much slower and needs substantial system RAM)
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo --local-device cpu
```

The first run downloads `mohammedaly22/QwenCleo-ASR` from Hugging Face. QwenCleo does not perform speaker diarization. Its timestamps mark approximate window starts, and exact repeated phrases from the overlapping audio are removed when they match.

## Notes

- Gemini defaults to `gemini-3.5-transcribe`. Speechmatics defaults to `melia-1` with `language=multi`. QwenCleo defaults to `mohammedaly22/QwenCleo-ASR` with the `Arabic` language hint.
- `--model` is provider-specific. Speechmatics accepts `melia-1`, `enhanced`, or `standard`.
- QwenCleo automatically uses short local windows, so do not combine it with `--chunk-minutes`; use `--local-chunk-seconds` instead.
- Speechmatics results are formatted with timestamps roughly every 90 seconds and whenever the detected speaker changes.
- If the output file already exists, choose a new filename with `-o`.
- Each chunk is saved as soon as it finishes. If a run stops, completed chunks remain in the output with an `INCOMPLETE` marker. Restarting begins from the start; it does not automatically resume.
- Timestamps and numbered speaker labels are local to each chunk. Each chunk heading shows its approximate start time in the full meeting.
- The Gemini `PROMPT` is written for Egyptian Arabic and English. Adjust it for meetings in other languages.
- Gemini `429` and model `404` errors can be checked in [AI Studio](https://aistudio.google.com/rate-limit). Speechmatics account usage and keys are managed in the [Speechmatics Portal](https://portal.speechmatics.com/).
