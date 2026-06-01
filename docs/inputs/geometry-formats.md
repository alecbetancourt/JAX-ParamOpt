# Geometry Formats

JAX-ParamOpt currently supports multiple geometry-loading pathways depending on backend and file format.

## Current Formats

- ReaxFF-style multi-structure BGF / extensionless geometry files
- HDF5 geometry containers

## Planned / Stubbed Formats

- XYZ
- PDB
- MOL2
- SDF

## Parser Contract

The geometry parser layer is intended to normalize formats into a common internal structure representation. When a source format supports it, a parser should ideally provide:

- structure or system name
- atom count
- atom names or types
- atomic numbers when available or inferable
- coordinates
- periodic box information
- total charge
- minimization or run metadata
- restraint data
- placeholder or embedded target arrays when relevant

Not every external format can provide all of these fields directly, so missing details may be populated with defaults and normalized later in the input pipeline.

## Canonical Path: BGF / Extensionless ReaxFF Geometry Files

The canonical geometry-loading path in the current codebase is the ReaxFF-style multi-structure BGF format handled by [`read_geo_file()`](/mnt/research/amberreax/betanc18/JAX-ParamOpt/jaxparamopt/input.py:123).

In practice, this parser accepts either:

- `.bgf` files
- extensionless geometry files written in the same multi-structure BGF-like layout

This path is currently the best-tested geometry input route and is the format used by the ReaxFF regression tests under [`tests/data/`](/mnt/research/amberreax/betanc18/JAX-ParamOpt/tests/data).

### File Structure

A single geometry file may contain many structures concatenated together. Each structure is delimited by `END` and typically contains:

- a `DESCRP` line naming the structure
- a `RUTYPE` line indicating whether the structure is a single-point or minimization case
- one or more `HETATM` records for atomic coordinates
- optional `CRYSTX` periodic box information
- optional `MOLCHARGE` total charge information
- optional restraint records
- optional `CONECT` records, which are currently not required by the parser

Examples of this layout can be seen in:

- [`tests/data/disulfide/geo`](/mnt/research/amberreax/betanc18/JAX-ParamOpt/tests/data/disulfide/geo:1)
- [`tests/data/silica/geo`](/mnt/research/amberreax/betanc18/JAX-ParamOpt/tests/data/silica/geo:1)

### Parsed Fields

For each structure in the file, the current parser constructs an internal `Structure` object containing:

- structure name from `DESCRP`
- atom count inferred from the number of `HETATM` records
- atom types derived from the backend-specific `name_to_index_map`
- approximate atomic numbers where they are inferred by the current loader
- Cartesian positions from `HETATM`
- periodic box information from `CRYSTX`, or a large nonperiodic default box otherwise
- total system charge from `MOLCHARGE` when present
- minimization flags derived from `RUTYPE`
- periodic image shifts computed from the box and cutoff
- optional bond, angle, and torsion restraints

The BGF parser currently uses dummy target arrays for embedded energy, force, and charge values. Training data is handled separately rather than being fully embedded in the geometry objects.

### Periodic and Nonperiodic Cases

If `CRYSTX` is present, the structure is treated as periodic and an orthogonalization matrix plus image shifts are constructed for later interaction-list generation.

If `CRYSTX` is absent, the structure is treated as nonperiodic and a large default box is used internally so the downstream distance machinery can still operate.

### Restraints

The parser currently recognizes:

- `BOND RESTRAINT`
- `ANGLE RESTRAINT`
- `TORSION RESTRAINT`

When restraints are absent, the loader inserts dummy placeholder restraint arrays so downstream code can operate on a consistent structure layout.

### Backend Dependence

Although the file layout itself is fairly generic, the current loader still contains backend-dependent atom typing logic:

- `reaxff` maps atom names through the ReaxFF forcefield `name_to_index` map
- `amber` and `ambereem` currently use similar mapping logic, including support for list-based per-structure maps

So this file format is not yet fully backend-neutral in implementation, even though the raw coordinate file could be reused across backends.

### Current Limitations

- Only a subset of element-to-atomic-number inference is implemented directly in the loader.
- `MOLCHARGE` support is currently limited to total system charge rather than arbitrary partial-charge assignments.
- `CONECT` records are present in many input examples but are not currently a primary source of topology in this parser.
- Embedded target values in the geometry file are not treated as the main training-data mechanism.
- The current parsing logic is inherited from the original ReaxFF workflow and will likely be normalized further as the input pipeline is refactored.

## HDF5 Geometry Containers

An HDF5 parser was added to support a more flexible container-based geometry path and to make it easier to store richer associated data such as Hessians.

At the moment, the HDF5 path should be viewed as a secondary, less mature input route relative to the BGF parser.

The current implementation expects per-system groups containing:

- `species`
- `coordinates`
- `nat`
- `structure_count`

and it skips a top-level `training` group if present.

The HDF5 path is intended to grow into a more general structured container format for:

- batched structures
- embedded training targets
- force data
- Hessians
- other metadata needed by more advanced workflows

## Planned / Stubbed Formats

The geometry dispatcher currently has explicit stubs for:

- `.xyz`
- `.pdb`
- `.mol2`
- `.sdf`

These are not implemented yet and currently raise `NotImplementedError`.

## Notes

HDF5 should be treated as a flexible container format rather than just a geometry format, since it may also carry training targets and higher-order data such as Hessians.
