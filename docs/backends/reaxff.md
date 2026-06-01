# ReaxFF Backend

The ReaxFF backend is currently the most complete path in the codebase and the main reference point for the ongoing refactor.

## Current Coverage

- forcefield loading
- parameter mapping
- geometry loading
- training data loading
- interaction list generation
- energy/force regression testing

## Current End-to-End Shape

The current ReaxFF path is the most representative “working” backend route in JAX-ParamOpt. In broad terms, it does the following:

1. load a ReaxFF forcefield file
2. load the parameter-selection file
3. map selected parameter entries onto forcefield indices
4. load geometries
5. load training data
6. filter and normalize those inputs
7. build interaction data and evaluate energies and forces

Much of the current backend refactor is using this path as the baseline that other backends should eventually match at the interface level, even if their internals differ substantially.

## Canonical Geometry Path

The canonical geometry path for ReaxFF is currently the multi-structure BGF / extensionless geometry file format described in [`Geometry Formats`](../inputs/geometry-formats.md).

This is the path used by the regression tests in:

- [`tests/reaxff_energy_test.py`](/mnt/research/amberreax/betanc18/JAX-ParamOpt/tests/reaxff_energy_test.py:1)
- [`tests/data/disulfide/`](/mnt/research/amberreax/betanc18/JAX-ParamOpt/tests/data/disulfide)
- [`tests/data/silica/`](/mnt/research/amberreax/betanc18/JAX-ParamOpt/tests/data/silica)

The current loader:

- reads many structures from a single file
- maps atom labels to ReaxFF atom-type indices using the loaded forcefield
- infers a limited set of atomic numbers directly in the parser
- preserves periodic-box and minimization metadata where present

## Forcefield Loading

The ReaxFF backend currently loads forcefields using the `jax-md` ReaxFF implementation and then fills derived forcefield tables such as:

- off-diagonal terms
- symmetry-expanded terms

This happens before parameter mapping and before the regression tests evaluate energies.

## Parameter Mapping

Parameter selection currently proceeds through a parameter file that specifies which entries of the forcefield should be optimized.

The current mapping logic supports:

- single-forcefield parameter entries
- grouped parameter entries
- multi-forcefield-style selectors, even though the present ReaxFF path usually operates on one forcefield at a time

This is one of the places where the code is already broader than the original single-use ReaxFF workflow, even if the interfaces are still rough.

## Training Data

The ReaxFF path currently supports both:

- text-based training data files
- HDF5 training data containers

Training data is loaded separately from the canonical BGF geometry file and then normalized into internal training-item structures. Geometry objects themselves currently carry placeholder target arrays rather than being the full source of training truth.

## Regression Testing

The most important current source of confidence for the ReaxFF backend is the regression test suite derived from the original JAX-ReaxFF work.

The renamed regression test:

- checks interaction-list size counts
- checks energies
- checks forces
- exercises the current clustering, alignment, and batched interaction allocation path

That makes the ReaxFF backend the best candidate for incremental refactoring, because it already has a concrete behavioral target.

## Current Limitations

- The backend is still tightly coupled to parts of the legacy workflow structure.
- Some import paths and runtime assumptions have needed adaptation to upstream `jax-md` changes.
- Input loading is only partially segmented from backend interpretation.
- HDF5 support exists but is not yet as mature or well-specified as the BGF path.
- Public API boundaries for ReaxFF-specific functionality are not final.

## Refactor Role

The ReaxFF backend should be treated as the first backend to fully normalize around the new architecture:

- backend-aware input loading
- stable workflow handoff objects
- explicit backend contracts
- narrower forcefield-specific modules

The goal is not just to preserve ReaxFF support, but to use it as the first fully documented and regression-tested backend template for broader forcefield support.

## Notes

This page should eventually document the precise ReaxFF assumptions inherited from the original JAX-ReaxFF workflow and what has changed in JAX-ParamOpt.
