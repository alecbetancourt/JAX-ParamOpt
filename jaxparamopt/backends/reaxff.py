"""ReaxFF backend input-loading support."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from jaxparamopt.config import OptimizationConfig

if TYPE_CHECKING:
  from jaxparamopt.input import InputBundle


class ReaxFFInputLoader:
  name = "reaxff"

  def load_inputs(self, config: OptimizationConfig) -> "InputBundle":
    import jax.numpy as jnp
    from jax_md import dataclasses
    from jax_md.mm_forcefields.reaxff.reaxff_forcefield import ForceField
    from jax_md.mm_forcefields.reaxff.reaxff_helper import read_force_field

    from jaxparamopt.input import (
      InputBundle,
      ParameterInput,
      create_structure_map,
      filter_data,
      map_params,
      read_geo_file,
      read_parameter_file,
      read_train_set,
      structure_training_data,
    )

    type_dtype = jnp.float64

    force_field = read_force_field(
      config.init_FF,
      cutoff2=config.cutoff2,
      dtype=type_dtype,
    )
    force_field = ForceField.fill_off_diag(force_field)
    force_field = ForceField.fill_symm(force_field)

    params_list_orig = read_parameter_file(config.params, ignore_sensitivity=0)
    params_list = map_params(params_list_orig, force_field.params_to_indices)

    param_indices = tuple(par[0] for par in params_list)
    bounds = tuple((par[2], par[3]) for par in params_list)

    systems = read_geo_file(config.geo, force_field.name_to_index, config.ff_type)

    if os.path.splitext(config.geo)[-1] in [".h5", ".hdf5"]:
      training_data = read_train_set(config.geo)
    else:
      training_data = read_train_set(config.train_file)

    validation_data = None
    systems, training_data = filter_data(systems, training_data)

    if config.use_valid:
      raise NotImplementedError("Validation data is not yet supported for ReaxFF input loading.")

    geo_name_to_index, geo_index_to_name = create_structure_map(systems)
    training_data = structure_training_data(training_data, geo_name_to_index)

    for index, system in enumerate(systems):
      systems[index] = dataclasses.replace(
        system,
        name=geo_name_to_index[system.name],
      )

    return InputBundle(
      config=config,
      backend_name=self.name,
      raw_model=force_field,
      raw_systems=systems,
      training_data=training_data,
      validation_data=validation_data,
      parameter_input=ParameterInput(
        raw_parameters=params_list,
        parameter_ids=param_indices,
        bounds=bounds,
        metadata={"raw_parameters_original": params_list_orig},
      ),
      metadata={
        "ffq_ff": None,
        "geo_name_to_index": geo_name_to_index,
        "geo_index_to_name": geo_index_to_name,
      },
    )
