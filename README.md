# TurboPaint

TurboPaint is a GPU-accelerated real-time AI image generation project. The
validated baseline uses PyTorch/Diffusers with SDXL-Turbo. The current stage
adds fixed-shape ONNX export and numerical validation as preparation for a
TensorRT backend; no TensorRT engine or custom CUDA backend exists yet.

## Repository layout

```text
src/
  backends/    Shared inference interface and generation configuration
  baseline/   PyTorch + Diffusers inference backend
  benchmark/  CUDA timing, statistics, configuration, and JSON reporting
  tensorrt/   ONNX export contracts/tools and future TensorRT backend
  cuda/       Reserved for future custom CUDA kernels
outputs/      Generated images (ignored by Git)
benchmarks/   Structured benchmark JSON (ignored by Git)
artifacts/    Exported ONNX graphs and validation reports (ignored by Git)
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

## ONNX export stage

The export boundary targets the two substantial image-generation networks, not
the complete Python pipeline:

- **Conditional UNet:** exported at batch 1 and fixed 512x512 image resolution
  (64x64 latent resolution). Its floating inputs and output are FP16.
- **VAE decoder:** exported at the same fixed latent/image resolution. The stock
  SDXL VAE declares `force_upcast=True`, so this graph deliberately uses FP32 to
  match Diffusers and avoid half-precision overflow. It must not be reported as
  an FP16 result.

Tokenizers, both CLIP text encoders, scheduler setup/step math, random latent
creation, and PIL post-processing remain in PyTorch/Diffusers. With one
denoising step, prompt encoding happens once and is not the primary optimization
target. Keeping scheduler and orchestration outside ONNX also lets the future
backend implement the existing `InferenceBackend` contract cleanly.

### Fixed tensor contracts

| Graph | Direction | Name | Shape | Dtype |
|---|---|---|---|---|
| UNet | input | `sample` | `[1, 4, 64, 64]` | `float16` |
| UNet | input | `timestep` | `[1]` | `float32` |
| UNet | input | `encoder_hidden_states` | `[1, 77, 2048]` | `float16` |
| UNet | input | `text_embeds` | `[1, 1280]` | `float16` |
| UNet | input | `time_ids` | `[1, 6]` | `float16` |
| UNet | output | `noise_pred` | `[1, 4, 64, 64]` | `float16` |
| VAE decoder | input | `latents` | `[1, 4, 64, 64]` | `float32` |
| VAE decoder | output | `images` | `[1, 3, 512, 512]` | `float32` |

The exports use ONNX opset 17, static shapes, external weight data for graphs
over ONNX's 2 GiB protobuf limit, and deterministic random validation tensors.
Each command runs the PyTorch component and ONNX Runtime CUDA execution with the
same tensors, checks numerical tolerance, and writes a `.validation.json`
report only on success.

Unsupported PyTorch-to-ONNX operators stop export. A neighboring
`.export_error.json` records the exception and tensor contract; do not use a
partially written graph or silently accept an unvalidated operator. An ONNX
Runtime CUDA provider/operator failure likewise stops numerical validation.

### Google Colab validation commands

Start a Colab runtime with a T4 GPU, then run these cells from a fresh session.
The ONNX Runtime pin below targets CUDA 12.x plus cuDNN 9, matching PyTorch 2.4+
CUDA 12 environments. The check must pass; if Colab changes its major runtime
versions, select the matching package from ONNX Runtime's official CUDA
compatibility table instead of forcing this pin.

```bash
!nvidia-smi
!git clone https://github.com/HetarthP/turbopaint.git
%cd turbopaint
!python -m src.runtime.validate
```

```python
import torch
print("torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
assert torch.cuda.is_available()
assert str(torch.version.cuda).startswith("12."), "Select a matching ORT build"
assert torch.backends.cudnn.version() // 10000 == 9, "Select a matching ORT build"
```

```bash
!pip install -r requirements-onnx.txt
!pip install onnxruntime-gpu==1.20.1
```

Restart the Colab runtime after installation if its preinstalled packages were
replaced, return to the cloned repository, then run:

```bash
%cd /content/turbopaint
!python -m src.runtime.validate
!python -m src.tensorrt.export.unet
!python -m src.tensorrt.export.vae_decoder
!find artifacts/onnx -maxdepth 3 -type f -printf '%p %k KiB\n'
!cat artifacts/onnx/unet/model.validation.json
!cat artifacts/onnx/vae_decoder/model.validation.json
```

These commands validate ONNX graphs only. They do not build TensorRT engines,
benchmark TensorRT, or establish a speedup.

## TensorRT engine build stage

The build stage consumes only the validated, fixed-shape ONNX graphs. It does
not implement pipeline inference yet.

- The UNet is strongly typed FP16 by its validated ONNX graph and retains the
  FP32 scheduler timestep input. TensorRT 11 removed the old weak-typing
  `BuilderFlag.FP16`; no obsolete precision flag is used.
- The VAE decoder remains strongly typed FP32 by its validated ONNX graph. No
  weak-typing or TF32 builder flag is used, preserving its safety contract.
- Both builders use batch 1 and the exact tensor names, shapes, and dtypes in
  the table above. A mismatch stops the build before serialization.
- The default TensorRT workspace limit is 4 GiB and can be changed with
  `--workspace-gib` if the target environment requires it.

Successful plans and metadata are cached at:

```text
artifacts/tensorrt/unet/model.plan
artifacts/tensorrt/unet/model.metadata.json
artifacts/tensorrt/vae_decoder/model.plan
artifacts/tensorrt/vae_decoder/model.metadata.json
```

Metadata records the TensorRT version, GPU name and compute capability,
precision policy, fixed input/output contract, ONNX bundle SHA-256 (including
external tensor data), workspace size,
and serialized engine size. A cache is reused only when those identity fields
match. TensorRT plans are not generally portable across TensorRT releases and
GPU architectures; use `--force` to rebuild deliberately.

Parser rejection prints every `OnnxParser` error with its index. Other build
failures retain TensorRT logger output and write a neighboring
`.build_error.json`. A failed or empty plan is never treated as a cache hit.

### Google Colab TensorRT build commands

Run these after both ONNX validation reports exist. First inspect Colab's CUDA
major version:

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
```

For a CUDA 12.x Colab runtime, install NVIDIA's full builder package and verify
that its Python builder initializes:

```bash
!python -m pip install --upgrade pip wheel
!python -m pip install --upgrade tensorrt-cu12
!python -c "import tensorrt as trt; print(trt.__version__); assert trt.Builder(trt.Logger())"
```

If the preceding Python check reports CUDA 13.x instead, install
`tensorrt-cu13`; do not mix CUDA-major package variants. The pip package is
enough for these Python builders and does not include `trtexec`.

Build each engine independently. `--verbose` is recommended for this first
validation run because it preserves detailed TensorRT layer/build logging:

```bash
%cd /content/turbopaint
!python -m src.tensorrt.build.unet --verbose
!python -m src.tensorrt.build.vae_decoder --verbose
```

Verify the serialized plans and inspect their real metadata:

```bash
!test -s artifacts/tensorrt/unet/model.plan
!test -s artifacts/tensorrt/vae_decoder/model.plan
!ls -lh artifacts/tensorrt/unet/model.plan artifacts/tensorrt/vae_decoder/model.plan
!cat artifacts/tensorrt/unet/model.metadata.json
!cat artifacts/tensorrt/vae_decoder/model.metadata.json
```

These commands build and cache engines only. They do not execute an end-to-end
TensorRT pipeline, benchmark it, or establish a speedup.

The builder uses `builder.create_network(0)`. TensorRT 11 is always explicit
batch and strongly typed, so the removed `EXPLICIT_BATCH` flag and deprecated
`STRONGLY_TYPED` flag are intentionally absent. The same zero-flags call is
valid on modern TensorRT 10 releases.

## Hybrid TensorRT inference

`SDXLTurboTensorRT` implements the same `InferenceBackend` contract as the
PyTorch baseline. Tokenization, both CLIP encoders, one-step Euler scheduler,
fixed-seed CUDA latent initialization, watermarking, and image post-processing
remain in Diffusers. Only the conditional UNet and VAE decoder calls use their
serialized TensorRT engines.

Each engine and execution context is deserialized once. TensorRT input bindings
point directly at CUDA PyTorch tensors on the current PyTorch stream, and output
buffers are persistent across generations. No NumPy/CPU transfer occurs between
the scheduler, UNet, and decoder. Because the validated decoder is FP32, one
persistent FP32 latent staging buffer is retained for its input.

Run end-to-end numerical validation before benchmarking. It generates the same
prompt and fixed seed with the PyTorch backend, releases it, then runs TensorRT
to avoid holding both complete implementations in T4 memory. Normalized RGB
images, error metrics, and both image files are stored under
`outputs/validation/`:

```bash
python -m src.tensorrt.validate \
  "A cinematic photograph of a robot painting a sunset" \
  --seed 0
```

Run the TensorRT smoke test (one warm-up plus one checked 512x512 image):

```bash
python -m src.tensorrt.smoke_test \
  "A cinematic photograph of a robot painting a sunset" \
  --seed 0
```

Run the comparable benchmark (five warm-ups, twenty CUDA-event measurements):

```bash
python -m src.tensorrt.benchmark \
  "A cinematic photograph of a robot painting a sunset" \
  --seed 0
```

The benchmark JSON records every latency, summary statistics, PyTorch allocator
peak, observed whole-device memory usage, and ratios against the supplied T4
PyTorch reference (610.60 ms mean, 654.38 ms p95, 1.64 images/s, 7.48 GiB peak
allocated VRAM). Whole-device memory includes other processes and is identified
separately. Ratios are measurements, not a speedup claim.

## Tests

```bash
pytest
```

The configuration, statistics, and report tests do not import PyTorch, download
the model, or require a GPU. CUDA execution must be validated separately on the
NVIDIA Linux machine.
