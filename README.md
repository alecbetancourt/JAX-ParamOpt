# WARNING: DEVELOPMENT IS CURRENTLY IN PROGRESS FOR A FIRST RELEASE, THIS README IS A WORK IN PROGRESS AND NO STABLE INTERFACE IS GUARANTEED

# JAX-ParamOpt

### Fast, flexible parameter optimization for molecular force fields - in pure Python, powered by JAX.

JAX-ParamOpt is a fast, flexible framework for optimizing parameters in molecular force fields. It combines differentiable energy evaluation with modern JAX tooling to deliver orders-of-magnitude speedups over traditional stochastic methods, while doubling as a sandbox for developing new functional forms and parameterization schemes in pure Python. This tool is a fork of the excellent JAX-ReaxFF optimizer by Mehmet Cagri Kaymak.

## Why JAX-ParamOpt?

- End-to-end differentiable: Uses JAX functional transformations to compute exact gradients of your loss w.r.t. any chosen parameters.

- Hardware acceleration: Scales from laptops to GPUs/TPUs with the same Python code.

- Minutes, not days: Gradient-based local optimizers converge dramatically faster than GA/MC for many fitting problems. In addition, a robust suite of GAs and other traditional optimization methods are supported to precondition

- General, not siloed: Works with ReaxFF, non-reactive force fields (e.g. AMBER Protein, GAFF), and is extensible to new terms.

## What’s new vs the original ReaxFF-only tool

- General parameter model: Optimize global, shared, and local parameters across one or many force-field objects.

- Clustered batching: Structures and FFs are aligned into clusters to enable vmap-friendly evaluation (1-FF/Many-geom or N-FF/N-geom) with minimal memory overhead.

- Fast set/get: Vectorized parameter scatter into FF arrays allows optimization fully in-memory, significantly reducing I/O pressure compared to other ad-hoc approaches.

- AMBER-family support (GAFF/FF19SB), ANI-1X support, easy to extend to new functional forms.

## Citations

You can learn more about the method in the following papers
(Plase cite them if you utlize this repository):

JAX-ParamOpt Paper: *In Progress*

Original JAX-ReaxFF Paper: [Jax-ReaxFF](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00363)

JAX-MD Integration Paper: [End-to-End Differentiable ReaxFF](https://link.springer.com/chapter/10.1007/978-3-031-32041-5_11)

In addition, cite these papers for some of the external dependencies that have been included in the core optimizer:
*In Progress*
DLFind
Sella
ANI
AMBER
Evosax
etc.

## How to Install
JAX-ParamOpt now uses `pyproject.toml` for package metadata and dependency groups.
The core package metadata is intentionally lighter than the full scientific stack so
that backend-specific dependencies can be installed explicitly.

**1-** Clone the repository:
```
git clone https://github.com/alecbetancourt/JAX-ParamOpt/
cd JAX-ParamOpt
```

**2-** Create an environment:
```
conda create -n jax-env python
conda activate jax-env
```

**3-** Install the package metadata and base dependencies:
```
pip install .
```

**4-** Install the JAX backend dependencies required for the current optimizer code paths:
```
pip install ".[jax]"
```

**5-** Optional dependency groups:
```
pip install ".[test]"        # pytest, coverage, test helpers
pip install ".[lint]"        # ruff, pre-commit
pip install ".[amber]"       # OpenMM / ParmEd support
pip install ".[global-opt]"  # evosax-based global optimization
pip install ".[dlfind]"      # DL-FIND integration
```

After setup, JAX-ParamOpt can be accessed via the command line interface with `jaxparamopt`.

#### Supported Install Modes

- `pip install .` installs the base package metadata and lightweight shared dependencies.
- `pip install ".[jax]"` installs the JAX runtime dependencies required for the current optimizer code paths.
- `pip install ".[amber]"`, `".[global-opt]"`, and `".[dlfind]"` add optional backend- or workflow-specific dependencies.
- `pip install ".[test]"` and `pip install ".[lint]"` install the local development tooling used for tests and quality checks.

#### API Stability Note

The CLI entrypoint is currently the primary supported interface.
The Python import surface is still provisional and may change as the backend and optimizer internals are refactored for the first public release.

**6-** GPU support is still handled through the JAX installation you choose. One example is:
```
pip install -U "jax[cuda12]==0.4.30"
```

To test the installation on a CPU (The JIT compilation time for CPUs drastically higher):
```
jaxparamopt --init_FF Datasets/cobalt/ffield_lit             \
            --params Datasets/cobalt/params                  \
            --geo Datasets/cobalt/geo                        \
            --train_file Datasets/cobalt/trainset.in         \
            --num_e_minim_steps 200                          \
            --e_minim_LR 1e-3                                \
            --out_folder ffields                             \
            --save_opt all                                   \
            --num_trials 1                                   \
            --num_steps 20                                   \
            --init_FF_type fixed                             
```          
You can learn more about JAX installation here: [JAX install guide](https://github.com/google/jax#installation)<br>

After installing the GPU version, the script will automatically utilize the GPU. If the script does not detect the GPU, it will print a warning message.


#### Using Validation Data
```
jaxparamopt --init_FF Datasets/disulfide/ffield_lit             \
            --params Datasets/disulfide/params                  \
            --geo Datasets/disulfide/geo                        \
            --train_file Datasets/disulfide/trainset.in         \
            --use_valid True                                    \
            --valid_file Datasets/disulfide/valSet/trainset.in  \
            --valid_geo_file Datasets/disulfide/valSet/geo      \
            --num_e_minim_steps 200                             \
            --e_minim_LR 1e-3                                   \
            --out_folder ffields                                \
            --save_opt all                                      \
            --num_trials 1                                      \
            --num_steps 20                                      \
            --init_FF_type fixed                             
``` 

#### Additional Documentation

*In Progress*

#### Potential Issues

On a HPC cluster, CUDA might be loaded somewhere different than /usr/local/cuda-xx.x. In this case, XLA compiler might not locate CUDA installation. This only happens if you install JAX with local CUDA support.
To solve this, we can speficy the cuda directory using XLA_FLAGS:
```
# To see where cuda is installed
which nvcc # will print /opt/software/CUDAcore/11.1.1/bin/nvcc
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_cuda_data_dir=/opt/software/CUDAcore/11.1.1"
```

Another potential issue related XLA compilation on clusters is *RuntimeError: Unknown: no kernel image is available for execution on the device* (potentially related to singularity)
and it can be solved by changing XLA_FLAGS to:

```
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_force_compilation_parallelism=1"
```
This flag can increase the compilation time drastically.
