# Testing

Testing is being expanded incrementally alongside the refactor.

## Current Layers

- lightweight interface tests
- ReaxFF energy/force regression tests
- future backend-specific integration tests

## Expected Local Commands

```bash
python -m pytest tests/config_test.py tests/driver_test.py tests/input_test.py
python -m pytest tests/reaxff_energy_test.py
```

## Future Direction

The test suite should eventually distinguish between:

- lightweight packaging and interface tests
- backend-specific runtime tests
- workflow-level regression tests
