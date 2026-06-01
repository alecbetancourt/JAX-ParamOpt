# Installation

JAX-ParamOpt uses `pyproject.toml` for packaging and optional dependency groups.

## Base Install

```bash
pip install jaxparamopt
```

Or from source:

```bash
git clone https://github.com/alecbetancourt/JAX-ParamOpt.git
cd JAX-ParamOpt
pip install .
```

## Optional Dependency Groups

```bash
pip install ".[jax]"
pip install ".[amber]"
pip install ".[global-opt]"
pip install ".[dlfind]"
pip install ".[test]"
pip install ".[lint]"
pip install ".[docs]"
```

## Recommended Environments

- `uv` or `venv` for lightweight packaging, lint, and docs work.
- `conda` or `mamba` for scientific stacks that include heavier binary dependencies.

## Accelerator Support

JAX-ParamOpt relies on the local JAX installation for CPU and accelerator support. Follow the official JAX installation guide for CPU, CUDA, or other accelerator-specific environments.
