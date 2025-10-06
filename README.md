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
JAX-ParamOpt requires JAX and jaxlib ([Jax Repo](https://github.com/google/jax)). <br>
The code is tested with JAX 0.4.26 - 0.4.30 and jaxlib 0.4.26 - 0.4.30.
Since the optimizer is highly more performant on GPUs, GPU version of jaxlib needs to be installed (GPU version supports both CPU and GPU execution). <br>

**1-** Before the installation, a supported version of CUDA and CuDNN are needed (for jaxlib). Alternatively, one could install the jax-md version that comes with required CUDA libraries. <br>

**2-** Cloning the JAX-ParamOpt repo:
```
git clone https://github.com/alecbetancourt/JAX-ParamOpt/
cd JAX-ParamOpt
```

**3-** The installation can be done in a conda environment:
```
conda create -n jax-env python=3.10
conda activate jax-env
```
**4-** JAX-ParamOpt is installed with the following command:
```
pip install .
```
After the setup, Jax-ReaxFF can be accessed via command line interface(CLI) with **jaxparamopt**

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
**5-** To have the GPU support, jaxlib with CUDA support needs to be installed, otherwise the code can only run on CPUs.
```
pip install -U "jax[cuda12]==0.4.30"
```
You can learn more about JAX installation here: [JAX install guide](https://github.com/google/jax#installation)<br>

After installing the GPU version, the script will automatically utilize the GPU. If the script does not detect the GPU, it will print a warning message.


#### Using Validation Data
```
jaxreaxff --init_FF Datasets/disulfide/ffield_lit             \
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
