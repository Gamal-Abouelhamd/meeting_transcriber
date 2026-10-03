# Meeting Transcription with Gemini 3, Speechmatics, or QwenCleo

A Python script that transcribes meeting recordings while preserving Egyptian Arabic and English as spoken. Gemini 3 Flash Preview (`gemini-3-flash-preview`) is the default model; Speechmatics and fully local QwenCleo-ASR are available through the same command-line interface.

## 1. Install dependencies

Use Python 3.10 or newer. For Gemini and Speechmatics, open a terminal in the project folder and run:

```powershell
python -m pip install -r requirements.txt
```

Splitting recordings with `--chunk-minutes` and all local QwenCleo transcription require **ffmpeg** on your `PATH`. It is a separate program, not a Python dependency. Check that it is installed:

```powershell
ffmpeg -version
```

If it is missing, download it from the [ffmpeg website](https://ffmpeg.org/download.html), add its `bin` folder to `PATH`, and open a new terminal.

## 2. Set up your API key once (cloud providers)

For Gemini or Speechmatics, copy the environment template if you do not already have a `.env` file:

```powershell
Copy-Item .env.example .env
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey), the [Speechmatics Portal](https://portal.speechmatics.com/), or both. Open `.env` next to `transcribe.py` and paste each key you plan to use after `=`:

```dotenv
GEMINI_API_KEY=your_api_key_here
SPEECHMATICS_API_KEY=your_api_key_here
```

Save the file. The script loads it automatically using `python-dotenv`, even when launched from another folder. You do not need to enter the key in the terminal each time. QwenCleo does not require this step.

If a key is already set in your terminal environment, it takes precedence over `.env`. Update or remove that environment variable if it contains an outdated value.

The `.gitignore` file excludes `.env`, local environment variants, recordings, and `.txt` output files. The `.env.example` template contains no key and can be shared in Git.

## 3. Transcribe a meeting

### Gemini 3 Flash Preview (default)

The script uses `gemini-3-flash-preview` automatically. Replace the filename with your recording's name, or provide its full path in quotes:

```powershell
python transcribe.py "meeting.mp4" --chunk-minutes 20
```

This splits the audio into chunks of approximately 20 minutes and saves the transcript next to the recording as `meeting.transcript.txt`.

To specify the same model explicitly:

```powershell
python transcribe.py "meeting.mp4" --model gemini-3-flash-preview --chunk-minutes 20
```

For a short recording, you can omit `--chunk-minutes` to transcribe the whole file in one request. Long recordings can exceed the transcript output limit even when the input fits; reduce the chunk size if that happens.

To choose the output filename or location:

```powershell
python transcribe.py "D:\Meetings\weekly meeting.mp3" --chunk-minutes 20 -o "weekly.transcript.txt"
```

### Speechmatics

Choose Speechmatics with `--provider speechmatics`:

```powershell
python transcribe.py "meeting.mp4" --provider speechmatics
```

The script defaults to the `melia-1` model with `language=multi`, Arabic and English hints (`ar,en`), and speaker diarization. Splitting is optional. To set the language hints explicitly:

```powershell
python transcribe.py "meeting.mp4" --provider speechmatics --language-hints ar,en
```

To use a single-language Speechmatics model instead:

```powershell
python transcribe.py "meeting.mp4" --provider speechmatics --model enhanced --language ar
```

### QwenCleo-ASR (local, no API key)

QwenCleo requires a separately prepared Python environment containing `qwencleo_asr` and PyTorch, plus ffmpeg on your `PATH`. These Python packages are not installed by `requirements.txt`, and this repository does not include the `install_qwencleo.ps1` helper referenced by the script's error messages.

If you already have that environment at `.venv-qwencleo`, transcribe locally from Windows PowerShell:

```powershell
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo
```

To use longer windows and a larger token limit:

```powershell
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" `
  --provider qwencleo `
  --local-chunk-seconds 60 `
  --local-overlap-seconds 5 `
  --local-max-new-tokens 768 `
  -o "meeting.60s.transcript.txt"
```

The local provider defaults to 20-second windows, 2 seconds of overlap, and 256 new tokens per window. Device and numeric type are selected automatically: CUDA when available, otherwise CPU; BF16 on supported CUDA hardware, FP16 on other CUDA hardware, and FP32 on CPU. QwenCleo keeps the model loaded while processing all windows. No recording is uploaded and no API key is needed.

Useful overrides:

```powershell
# Automatic language detection instead of the default Arabic hint
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo --language auto

# Smaller windows if CUDA runs out of memory
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo --local-chunk-seconds 15 --local-overlap-seconds 2

# CPU fallback (much slower and needs substantial system RAM)
.\.venv-qwencleo\Scripts\python.exe .\transcribe.py "meeting.mp4" --provider qwencleo --local-device cpu
```

The first run downloads `mohammedaly22/QwenCleo-ASR` if it is not already cached. QwenCleo does not perform speaker diarization. Its timestamps mark approximate window starts in the full recording, and matching repeated phrases from overlapping audio are removed.

## Notes

- Gemini defaults to `gemini-3-flash-preview`. Speechmatics defaults to `melia-1` with `language=multi`. QwenCleo defaults to `mohammedaly22/QwenCleo-ASR` with the `Arabic` language hint.
- `--model` is provider-specific: a Gemini model ID, a Speechmatics model (`melia-1`, `enhanced`, or `standard`), or a QwenCleo checkpoint.
- `--language` applies to Speechmatics and QwenCleo. `--language-hints` applies to Speechmatics `melia-1`. For Gemini, adjust `PROMPT` in `transcribe.py` to change the expected languages.
- QwenCleo automatically uses short local windows, so do not combine it with `--chunk-minutes`; use `--local-chunk-seconds` instead.
- Speechmatics results are formatted with timestamps roughly every 90 seconds and whenever the detected speaker changes.
- If the output file already exists, choose a new filename with `-o`.
- With `--chunk-minutes` or local QwenCleo windows, each completed chunk/window is saved immediately. If a run stops, completed text remains in the output with an `INCOMPLETE` marker. Restarting begins from the start; choose a new output filename because the script does not overwrite existing files or resume automatically.
- Cloud runs without `--chunk-minutes` write the output only after the whole recording succeeds.
- With `--chunk-minutes`, timestamps and numbered speaker labels are local to each cloud chunk. Each chunk heading shows its approximate start time in the full meeting. QwenCleo timestamps refer to window starts in the full recording.
- Check Gemini quota and usage in [AI Studio](https://aistudio.google.com/rate-limit). For a model error, check the model ID passed to `--model` and your account's access. Speechmatics account usage and keys are managed in the [Speechmatics Portal](https://portal.speechmatics.com/).
