# ljudanteckning

Make your media library searchable: scan folders, split audio, transcribe in parallel across NVIDIA GPUs, then export subtitles and searchable text next to each original file.

## Goals

- Recursively discover media files (audio + video)
- Extract audio and chunk into short WAV segments (FFmpeg)
- Transcribe chunks in parallel across N GPUs (faster-whisper / CTranslate2)
- Merge timestamps correctly and export:
  - `.srt`
  - `.vtt`
  - `.whisper.json`
  - `.txt`
- Clean up temp files after success
- Nice terminal UX:
  - colored output + banner
  - interactive TUI mode (later)
  - batch mode (`--nocli`)

## Requirements

Ljudanteckning is designed for **NVIDIA GPUs** and gets its best performance with **CUDA**.
If you have multiple GPUs available, it can run several transcription workers in parallel.

### Core requirements
- Linux
- Python 3.10+
- FFmpeg (`ffmpeg` and `ffprobe`)

### GPU requirements (recommended / expected)
- One or more **NVIDIA GPUs**
- NVIDIA proprietary driver installed (so `nvidia-smi` works)
- CUDA runtime libraries compatible with your driver  
  (installing the CUDA toolkit/runtime is often the simplest way to get these)

### Optional
- NVML Python bindings for nicer GPU telemetry in the UI (`pip install -e ".[nvml]"`)

### Quick check
```bash
nvidia-smi
```
If that command works and your GPUs show up, you’re ready to run GPU-accelerated transcription.

## Screenshots

Here’s what a real run looks like on a multi-GPU box: chunking with FFmpeg, parallel transcription workers, and live GPU telemetry.

### Multi-GPU transcription + live telemetry (the fun part)
<img src="docs/screenshots/02-transcribing.png" width="900" alt="Ljudanteckning transcribing chunks across multiple NVIDIA GPUs with live telemetry." />

<details>
  <summary>More screenshots</summary>

  <p><strong>Chunking stage</strong></p>
  <img src="docs/screenshots/01-chunking.png" width="900" alt="Ljudanteckning chunking audio using FFmpeg." />

  <p><strong>Finished outputs next to media</strong></p>
  <img src="docs/screenshots/03-finished.png" width="900" alt="Ljudanteckning finished run showing exported subtitles and transcript files next to media." />
</details>


## Install (developer setup)

### Arch Linux (fish)
```fish
# 1) System packages
sudo pacman -Syu
sudo pacman -S --needed git ffmpeg python python-pip nvidia-utils

# 2) GPU sanity check
# If this fails, fix your NVIDIA driver installation first.
nvidia-smi

# 3) Clone + virtualenv + editable install
git clone https://github.com/albinhenriksson/ljudanteckning.git
cd ljudanteckning

python -m venv .venv
source .venv/bin/activate.fish

python -m pip install -U pip
pip install -e ".[dev,nvml]"

ljudanteckning --help
```

### Debian / Ubuntu (bash)
```bash
# 1) System packages
sudo apt-get update
sudo apt-get install -y git ffmpeg python3 python3-venv python3-pip

# 2) GPU sanity check
# If `nvidia-smi` is not found, install the NVIDIA driver/utils for your distro first.
nvidia-smi

# 3) Clone + virtualenv + editable install
git clone https://github.com/albinhenriksson/ljudanteckning.git
cd ljudanteckning

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev,nvml]"

ljudanteckning --help
```

### Fedora (bash)
```bash
# 1) System packages
sudo dnf install -y git ffmpeg python3 python3-pip

# 2) GPU sanity check
# If `nvidia-smi` is not found, install the NVIDIA driver/utils for your distro first.
nvidia-smi

# 3) Clone + virtualenv + editable install
git clone https://github.com/albinhenriksson/ljudanteckning.git
cd ljudanteckning

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev,nvml]"

ljudanteckning --help
```

## Quick start

> Note: CLI flags shown below are implemented progressively while the pipeline is being built.
> Run `ljudanteckning run --help` to see what’s available in your installed version.

### 1) Create a local config (recommended)
```bash
cp ljudanteckning.example.ini ljudanteckning.ini
```

(Optional) sanity-check what Ljudanteckning will use:
```bash
ljudanteckning config
```

### 2) Transcribe a single file
```bash
ljudanteckning run --nocli "/mnt/media/Mr.Robot.S01E01.1080p.BluRay.mkv"
```

### 3) Transcribe an entire directory (recursive)
```bash
ljudanteckning run --nocli "/mnt/media"
```

### 4) Pick GPUs explicitly (multi-GPU)
Example with 4 GPUs:
```bash
ljudanteckning run --nocli --gpus "0,1,2,3" --jobs 4 "/mnt/media"
```

Example for an 11-GPU box:
```bash
ljudanteckning run --nocli --gpus "0,1,2,3,4,5,6,7,8,9,10" --jobs 11 "/mnt/media"
```

### 5) Override model / language
```bash
ljudanteckning run --nocli --model large-v3 --language sv "/mnt/media"
```

## Outputs

For each media file, Ljudanteckning writes transcript/subtitle files **next to the original**, e.g.:

- `Movie.mkv`
- `Movie.srt`
- `Movie.vtt`
- `Movie.txt`
- `Movie.whisper.json`

Temporary chunks and per-chunk JSON are stored in a hidden work folder next to the file (by default: `.ljudanteckning/`)
and are cleaned up according to `output.cleanup` in your INI.
