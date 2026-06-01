# JAX-ParamOpt

JAX-ParamOpt is a framework for optimizing molecular force-field parameters with differentiable energy evaluation and JAX-native execution. It began as an extension of the original JAX-ReaxFF optimizer developed by Memhet Cagri Kaymak and is being refactored into a broader forcefield-agnostic optimization platform.

## Documentation Structure

- `Installation` covers package setup, optional dependency groups, and environment choices.
- `Quickstart` provides the first CLI and Python entry points to learn the package.
- `CLI` and `Python API` describe the two public entry surfaces as they stabilize.
- `Backends` documents forcefield-specific capabilities and limitations.
- `Inputs` describes geometry, training, and parameter file formats.
- `Workflows` covers optimization task structure and staged fitting strategies.
- `Theory` contains conceptual and scientific documentation.
- `Reference` is reserved for stable API documentation.
- `Development` contains architecture, contributing, and testing notes.

## Current Status

- The CLI is the primary supported interface.
- The Python API is still provisional.
- Input loading and backend segmentation are under active refactor.
- Reference/API docs will expand as module boundaries stabilize.
