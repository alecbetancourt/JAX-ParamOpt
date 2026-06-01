# Inputs

Input handling is being refactored around a common scaffold:

- geometry or system inputs
- training and validation targets
- parameter selection and bounds
- backend-specific normalization

## Current Focus

The current codebase is converging on a backend-aware input pipeline while keeping raw readers reusable where possible.
