# WARNING: DEVELOPMENT IS CURRENTLY IN PROGRESS FOR A FIRST RELEASE, THIS README IS A WORK IN PROGRESS AND NO STABLE INTERFACE IS GUARANTEED

# JAX-ParamOpt

[![CI](https://github.com/alecbetancourt/JAX-ParamOpt/actions/workflows/ci.yml/badge.svg)](https://github.com/alecbetancourt/JAX-ParamOpt/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

<!-- Placeholder badges to enable after first release:
[![PyPI version](https://img.shields.io/pypi/v/jaxparamopt.svg)](https://pypi.org/project/jaxparamopt/)
[![Coverage](https://img.shields.io/codecov/c/github/alecbetancourt/JAX-ParamOpt)](https://codecov.io/gh/alecbetancourt/JAX-ParamOpt)
[![DOI](https://zenodo.org/badge/DOI/<doi>.svg)](https://doi.org/<doi>)
-->

### Fast, flexible parameter optimization for molecular force fields in pure Python, powered by JAX.

JAX-ParamOpt is a framework for optimizing molecular force-field parameters with differentiable energy evaluation and modern JAX tooling. It began as a fork of the JAX-ReaxFF optimizer by Mehmet Cagri Kaymak and is being generalized to support broader force-field parameter sets, multiple local and global optimization methods, and reusable library workflows in addition to a CLI.

## Why JAX-ParamOpt?

- End-to-end differentiable optimization of selected force-field parameters.
- Pure Python implementation with JAX transformations for CPU/GPU execution.
- Support for both gradient-based local optimization and more traditional global search methods.
- Generalization beyond ReaxFF toward broader force-field representations and interchange workflows.

## Status

- The CLI entrypoint is currently the primary supported interface.
- The Python API is still provisional and may change as the backend and workflow internals are refactored.
- The package metadata and CI/release scaffolding are in place, but the scientific workflow and test surface are still being stabilized for a first public release.

## Installation

JAX-ParamOpt uses `pyproject.toml` for packaging and optional dependency groups. The base package is intentionally lighter than the full scientific runtime so backend-specific dependencies can be installed explicitly.

### Install From PyPI

Once a public release is available:

```bash
pip install jaxparamopt
```

Optional dependency groups can then be installed as needed:

```bash
pip install "jaxparamopt[jax]"
pip install "jaxparamopt[test]"
pip install "jaxparamopt[lint]"
pip install "jaxparamopt[amber]"
pip install "jaxparamopt[global-opt]"
pip install "jaxparamopt[dlfind]"
```

### Install From Source

```bash
git clone https://github.com/alecbetancourt/JAX-ParamOpt.git
cd JAX-ParamOpt
pip install .
```

To include the current JAX runtime dependencies used by the optimizer code paths:

```bash
pip install ".[jax]"
```

### Supported Install Modes

- `pip install .` or `pip install jaxparamopt` installs the base package metadata and lightweight shared dependencies.
- `pip install ".[jax]"` installs JAX, `jaxlib`, and the current `jax-md` fork dependency used by the optimizer.
- `pip install ".[amber]"` adds OpenMM and ParmEd support.
- `pip install ".[global-opt]"` adds `evosax`-based global optimization dependencies.
- `pip install ".[dlfind]"` adds DL-FIND integration.
- `pip install ".[test]"` installs the current local test toolchain.
- `pip install ".[lint]"` installs `ruff` and `pre-commit`.

## Environment Setup

The package is published and tested as a standard Python package, but the recommended environment manager depends on what you are trying to do.

### Option 1: `uv` or `venv`

This is the closest match to the GitHub Actions runners.

Using `uv`:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install ".[test]"
python -m pytest tests/config_test.py tests/driver_test.py
```

Using the standard library `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[test]"
python -m pytest tests/config_test.py tests/driver_test.py
```

### Option 2: `conda` or `mamba`

This is a good choice if conda-exclusive downstream dependencies are required.

```bash
conda create -n jaxparamopt python=3.12
conda activate jaxparamopt
pip install .
pip install ".[jax]"
```

If you need optional backends:

```bash
pip install ".[amber,global-opt,dlfind]"
```

On HPC clusters or other managed systems, prefer a user-created environment and unload unrelated Python or scientific software modules before installing or running the package. In particular, avoid inheriting a site-managed `PYTHONPATH` or system Python module into a `conda`, `venv`, or `uv` environment, as that can cause package-resolution and permission conflicts.

### Option 3: install from a source archive or GitHub release artifact

Once release artifacts exist:

```bash
pip install jaxparamopt-<version>-py3-none-any.whl
```

or

```bash
pip install jaxparamopt-<version>.tar.gz
```

## JAX and Accelerator Support

JAX-ParamOpt relies on the JAX installation in your environment for CPU and accelerator support.

- CPU-only and NVIDIA GPU installs should follow the official JAX installation guide.
- GPU compatibility is intentionally handled downstream through `jax`/`jaxlib`, rather than through separate JAX-ParamOpt wheels.

See the official JAX installation instructions here:

- [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)

## Basic Usage

After installation, the CLI entrypoint is available as:

```bash
jaxparamopt --help
```

A small ReaxFF-style example:

```bash
jaxparamopt --init_FF Datasets/cobalt/ffield_lit \
            --params Datasets/cobalt/params \
            --geo Datasets/cobalt/geo \
            --train_file Datasets/cobalt/trainset.in \
            --num_e_minim_steps 200 \
            --e_minim_LR 1e-3 \
            --out_folder ffields \
            --save_opt all \
            --num_trials 1 \
            --num_steps 20 \
            --init_FF_type fixed
```

Validation-data support exists in the argument surface but is not yet a stable, release-ready workflow.

## Development

The repository currently uses:

- `pyproject.toml` for package metadata and tool configuration.
- `pre-commit` plus `ruff` for lightweight quality checks.
- GitHub Actions `ci.yml` for build smoke tests, linting, and the first interface tests.
- GitHub Actions `release.yml` for source/wheel builds and future release publishing.
- MkDocs plus GitHub Pages for the main documentation site.

Typical local commands:

```bash
python -m pip install ".[lint,test]"
pre-commit run --all-files
python -m pytest tests/config_test.py tests/driver_test.py
```

Documentation commands:

```bash
python -m pip install ".[docs]"
mkdocs serve
mkdocs build
```

## Configuration Notes

Several runtime behaviors are controlled outside the package itself through JAX/XLA configuration.

- Precision:
  The current driver enables `jax_enable_x64`, so double precision is expected by default in the current workflow.
- GPU installation:
  Follow the JAX installation guide for the correct CUDA-enabled `jaxlib` package rather than relying on package-local CUDA logic.
- Nonstandard CUDA locations on clusters:
  If XLA cannot find your CUDA installation, you may need:

```bash
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_cuda_data_dir=/path/to/cuda"
```

- Certain cluster or container setups may require:

```bash
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_force_compilation_parallelism=1"
```

  This can substantially increase compilation time.

- Memory fraction:
  The current driver sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` if it is not already defined in the environment.
- Managed-system Python conflicts:
  On HPC systems, starting from a clean shell can prevent conflicts with site-installed Python packages. Using `module purge` and unsetting `PYTHONPATH` before activating a user-managed environment is often helpful.

## Citations

If you use this repository, please cite the relevant upstream and project-specific work.

- JAX-ParamOpt paper: in progress
- Original JAX-ReaxFF paper: [JAX-ReaxFF](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00363)
- JAX-MD integration paper: [End-to-End Differentiable ReaxFF](https://link.springer.com/chapter/10.1007/978-3-031-32041-5_11)

Additional dependency-specific citation guidance will be added as the first release is prepared.
