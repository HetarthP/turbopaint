# TurboPaint

TurboPaint is a GPU-accelerated real-time AI image generation project. This
first milestone provides a modular PyTorch/Diffusers baseline for SDXL-Turbo;
TensorRT and custom CUDA backends are reserved for later milestones.

## Repository layout

```text
src/
  backends/    Shared inference interface and generation configuration
  baseline/   PyTorch + Diffusers inference backend
  benchmark/  CUDA timing, statistics, configuration, and JSON reporting
  tensorrt/   Reserved for a future TensorRT backend
  cuda/       Reserved for future custom CUDA kernels
outputs/      Generated images (ignored by Git)
benchmarks/   Structured benchmark JSON (ignored by Git)
tests/        Unit tests
```

## Requirements

- Python 3.10+
- An NVIDIA GPU with sufficient VRAM
- A compatible NVIDIA driver and CUDA-enabled PyTorch build
- Access to the `stabilityai/sdxl-turbo` model on Hugging Face

CUDA is required intentionally. The program exits with a clear error instead
of silently falling back to CPU, because CPU timings would not be comparable
with future optimized backends.

## Installation

`requirements.txt` deliberately does not contain `torch`: an unqualified
`pip install torch` can select the wrong build, and this repository cannot know
which CUDA wheel matches an unseen Linux host. Use Python 3.11 where practical,
create a virtual environment, and select **Linux / Pip / Python / the CUDA
version supported by the target driver** at the official
[PyTorch selector](https://pytorch.org/get-started/locally/).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Run the exact CUDA-enabled PyTorch command produced by the official selector.
# Do not substitute a CPU-only wheel and do not guess a CUDA index URL.

pip install -r requirements.txt
```

For development and unit tests, use `pip install -r requirements-dev.txt` after
installing the appropriate PyTorch build.

## GPU validation workflow

Run these commands from the repository root, in order.

### 1. Validate the environment

```bash
python -m src.runtime.validate
```

This prints Python and PyTorch versions, CUDA availability and runtime version,
GPU name/count, total memory, and PyTorch's currently allocated/reserved GPU
memory. It does not load SDXL-Turbo. A nonzero exit status means the machine is
not ready.

### 2. Run the smoke test

```bash
python -m src.baseline.smoke_test \
  "A cinematic photograph of a robot painting a sunset" \
  --seed 0
```

This loads SDXL-Turbo in FP16, performs exactly one warm-up and one 512x512
generation, saves the second image under `outputs/`, reopens it, and verifies
its dimensions. It does not run or write a benchmark.

### 3. Run the full baseline benchmark

```bash
python -m src.baseline.benchmark \
  "A cinematic photograph of a robot painting a sunset" \
  --seed 0
```

Defaults are a 512x512 image, 1 inference step, 5 warm-up runs, and 20 measured
runs. SDXL-Turbo is designed for very low step counts and guidance scale 0.0.
Useful options include:

```bash
python -m src.baseline.benchmark "your prompt" \
  --warmup-runs 5 \
  --benchmark-runs 20 \
  --seed 42 \
  --output outputs/result.png
```

Run `python -m src.baseline.benchmark --help` for every option.

The prompt, 512x512 resolution, inference steps, guidance scale, and fixed seed
remain identical for every warm-up and measured generation. The reported
latency covers GPU work submitted by the Diffusers pipeline. It excludes
one-time model loading and image file encoding. CUDA events are recorded around
each inference; TurboPaint synchronizes after warm-up and again before reading
the events. Every latency and its mean, median, p95, minimum, maximum, and
images/second are saved as structured JSON in `benchmarks/`. The last generated
image is saved in `outputs/`. The report also records PyTorch's peak allocated
GPU memory after resetting peak statistics immediately before measured runs.
This is allocator-visible memory, not whole-system or other-process GPU usage.

The backend is constructed before `benchmark_cuda` is called, so model download,
deserialization, and transfer to the GPU are excluded. PNG saving happens only
after all measurements and synchronization complete, so it is also excluded.

No result file is created unless a real CUDA benchmark completes successfully.

## NVIDIA Linux machine setup

Use a CUDA-capable Linux host for actual inference and performance collection.
The exact driver and PyTorch CUDA wheel should be chosen together rather than
copied from the macOS development environment.

1. Provision a supported NVIDIA GPU with enough VRAM for SDXL-Turbo and install
   a current NVIDIA Linux driver.
2. Verify the driver and visible GPU with `nvidia-smi`.
3. Install Python 3.11 (recommended) and create an isolated virtual environment.
4. From the official PyTorch installation selector, install the CUDA-enabled
   PyTorch wheel compatible with the host driver. Do not install a CPU-only
   wheel.
5. Install the repository dependencies with `pip install -r requirements.txt`.
6. If Hugging Face requests authentication, run `huggingface-cli login` with an
   account that can access `stabilityai/sdxl-turbo`.
7. Run the environment validation, smoke test, and full benchmark commands from
   the GPU validation workflow above, in that order.

For comparable runs, keep the software environment, GPU power/clock policy,
prompt, generation configuration, and benchmark configuration fixed. Record
the generated JSON alongside any future TensorRT results.

## Tests

```bash
pytest
```

The configuration, statistics, and report tests do not import PyTorch, download
the model, or require a GPU. CUDA execution must be validated separately on the
NVIDIA Linux machine.
