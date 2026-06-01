# Optimization Workflows

JAX-ParamOpt is moving toward a workflow-oriented architecture rather than a single monolithic optimization script.

## Intended Workflow Stages

- configuration normalization
- input loading
- dataset structuring
- backend preparation
- objective construction
- optimization execution
- output finalization

## Status

This page should be updated as the workflow orchestration is extracted from `driver.py` into clearer internal stages.
