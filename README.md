# ai-infra-minis

[![CI](https://github.com/emmanuel-schilling/ai-infra-minis/actions/workflows/ci.yml/badge.svg)](https://github.com/emmanuel-schilling/ai-infra-minis/actions/workflows/ci.yml)

Small, measured projects on how LLM inference works and where its time goes.
Each project builds one idea, tests it against a simple reference, and records
its benchmarks. Projects are added here as they start.

## Projects

| Project | What it explores | Language | Status |
|---|---|---|---|
| [Quantized matmul](projects/quant-matmul/) | Scalar, AVX2 + FMA and 8-bit quantized matrix multiply, compared with a vendor BLAS (Accelerate on macOS, OpenBLAS elsewhere) | C++20 | Active |

Each project folder has a `STATUS.md` (state, open questions, decisions and
their reasons) and a `RESULTS.md` (benchmarks and conclusions).

## How results are measured

- Release builds, scripted and repeatable. Each run records the hardware,
  compiler flags and exact command.
- Raw output goes in a new dated folder under the project's `results/`;
  earlier runs are never overwritten.
- Repeated runs with cooldowns, reported as medians, because the development
  laptop (Intel Core i9-8950HK, AVX2 and FMA) throttles under sustained load.
- Every optimized version is tested against a simple reference implementation
  before it is timed.

## Building and testing

The code builds and is tested on Linux, macOS and Windows. CI runs the steps
below on all three.

Requirements:
- CMake 4.0 or later, Ninja and a C++20 compiler
- [vcpkg](https://vcpkg.io/), with `VCPKG_ROOT` set to its checkout; it
  installs GoogleTest, Google Benchmark and, on Linux and Windows, OpenBLAS,
  at the versions pinned in `vcpkg.json`
- [uv](https://docs.astral.sh/uv/) for the Python tools

```sh
# C++: debug build with AddressSanitizer and UndefinedBehaviorSanitizer
cmake --preset dev
cmake --build --preset dev
ctest --preset dev

# Python: format, lint, type-check and test
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

## Running the benchmarks

```sh
uv run python projects/quant-matmul/scripts/run_bench.py
```

This builds the `release` preset, runs the benchmarks for several rounds with
cooldowns between them, and writes the raw output, the machine and build
details, and the medians to a new dated folder under
`projects/quant-matmul/results/`.

## License

[MIT](LICENSE)
