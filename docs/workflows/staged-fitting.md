# Staged Fitting

Many realistic forcefield optimization tasks are best expressed as staged workflows rather than single-pass fits.

## Examples

- static target fitting before simulation-derived targets
- local optimization after global initialization
- backend-specific prefit stages
- validation passes after parameter updates

## Notes

This page should eventually describe how staged workflows are represented in the package and how they map onto the Python and CLI interfaces.
