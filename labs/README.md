# Notebook labs

Write and check small, real Triton kernels in your own notebook runtime. Each lab has TODO exercises,
PyTorch correctness checks, and a complete Solution section after “try first”. Run cells in order;
starter failures are expected and print what is wrong. Run all continues to the solutions, checks them
strictly, and then measures them if a GPU is available. Each notebook stands alone and needs no other files
or API key. Backs lesson codes below identify the curriculum; those inference lessons are planned.

**Needs a free Google or Kaggle account; GPU optional.** Hosted runtimes and GPU availability are controlled
by those services. These links open their sites; the static Academy never runs or uploads your code.

| Lab | Backs | Open in your account |
| --- | --- | --- |
| [LAB-13 · First Triton kernel](lab-13-first-triton-kernel.ipynb) · ~20 min | I29, I33 | [Open in Colab](https://colab.research.google.com/github/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-13-first-triton-kernel.ipynb) · [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-13-first-triton-kernel.ipynb) |
| [LAB-14 · Fuse SiLU × multiply](lab-14-fuse-silu-mul.ipynb) · ~20 min | I28 | [Open in Colab](https://colab.research.google.com/github/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-14-fuse-silu-mul.ipynb) · [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-14-fuse-silu-mul.ipynb) |
| [LAB-15 · Fused softmax](lab-15-fused-softmax.ipynb) · ~25 min | I34 (also visual G10/G12) | [Open in Colab](https://colab.research.google.com/github/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-15-fused-softmax.ipynb) · [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-15-fused-softmax.ipynb) |
| [LAB-17 · Benchmark harness](lab-17-benchmark-harness.ipynb) · ~20 min | I35–I37 | [Open in Colab](https://colab.research.google.com/github/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-17-benchmark-harness.ipynb) · [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/atabak-eghbal/tile-and-error-labs/blob/main/labs/lab-17-benchmark-harness.ipynb) |

## Choose a GPU in Colab

1. Open a notebook using its Colab link and sign in. Save a copy in your Drive if you want to keep edits.
2. Open **Runtime → Change runtime type**.
3. Choose a **GPU** under Hardware accelerator, save, and connect to the runtime. Use a free option if available.
4. Run the setup cell first, then work downward. It prints the detected GPU and software versions.

## Choose a GPU in Kaggle

1. Open a notebook using its Kaggle link, sign in, and create/edit your own copy.
2. Open the notebook's **Settings** panel (sometimes under Notebook options).
3. Under **Accelerator**, choose an available free **GPU** and start the session. Follow any account verification prompt.
4. Run setup first, then the remaining cells. If Triton needs installation, the runtime must permit package installation.

The services may change their interface or have no free GPU available. You can still work through correctness
checks in a CPU runtime. No quota or guaranteed performance is assumed.

## CPU interpreter and measurements

Setup checks `torch.cuda.is_available()`. With no CUDA GPU, it sets `TRITON_INTERPRET=1` **before importing Triton**,
uses CPU tensors, and prints `No GPU: correctness checks only, timings skipped`. Triton's interpreter runs the
same kernels for correctness, not GPU performance. Use a fresh supported Linux Python runtime; native macOS is
not a supported Triton runtime. Colab/Kaggle include PyTorch; setup retries a missing Triton import after
`%pip install -q triton`. That installation happens in your notebook account, not in the Academy.

Restart the runtime before rerunning setup or switching devices: interpreter mode is chosen at the first Triton
import. No GPU timing code runs on CPU. LAB-17 still checks percentile arithmetic against PyTorch using an
explicitly illustrative distribution; CUDA-event harness checks wait for a GPU.

GPU measurements are labelled **your run**. Timed outputs are preallocated and the harness warms up before
collecting samples. Logical tensor bytes estimate effective GB/s; caches and runtime overhead mean this is not
physical memory-bus traffic. Only identified T4, P100 16 GB and H100 SXM variants show a datasheet-peak percentage;
unknown variants show GB/s without a percentage. A fixed speedup is never promised. LAB-17 measures an isolated
kernel, explaining which extra request measurements are needed for a serving load test.

Return to the [Academy Labs track](https://tile-and-error-academy.vercel.app/#labs) or the
[Tile & Error channel](https://www.youtube.com/@tileanderror).

## Local structure check (no network or dependencies)

From the repository root, run `python3 scripts/check_notebooks.py`, or `node scripts/check.mjs` to include the
existing JavaScript checks too. This validates cleared notebook structure, Python syntax (ignoring notebook
magic/shell lines), local notebook paths, Academy routes, and the exact Colab/Kaggle targets. It does not execute
Triton or prove that a hosted service can import an unpublished branch. Notebook links target `main` and work
after the files are published there.
