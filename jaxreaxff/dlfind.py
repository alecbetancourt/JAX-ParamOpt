import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORM_NAME"] = "cpu" # TODO: This may be deprecated
os.environ["JAX_PLATFORMS"] = "cpu" # Enforce JAX CPU-only backend

# TODO various methods exist to set this, inconsistent between backends
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
#os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import jax
import jax.numpy as jnp
import numpy as onp
import sys
import pickle
from flax.serialization import from_bytes
from libdlfind import dl_find
from libdlfind.callback import (dlf_get_gradient_wrapper,
                                dlf_put_coords_wrapper, make_dlf_get_params)
from jax_md.amber.amber_forcefield import AmberForceField, FFQForceField
from jax_md.amber.amber_energy_v2 import amber_energy
from jaxreaxff.structure import Structure, BondRestraint, AngleRestraint, TorsionRestraint
import functools
from frozendict import frozendict

@dlf_get_gradient_wrapper
def e_g_func(coordinates, iimage, kiter, ff, ffq_ff, nrg_fn):
  energy, grad = nrg_fn(coordinates, ff, None)
  return energy, grad

@dlf_put_coords_wrapper
def store_results(switch, energy, coordinates, iam, traj_coords, traj_energies):
  traj_coords.append(onp.array(coordinates))
  traj_energies.append(energy)
  return

def dlfind_min(pos, struct, ff, ffq_ff, max_iter):
  dlf_get_params = make_dlf_get_params(coords=pos, maxcycle=max_iter)

  nrg_fn, amber_ff, body_fn, state = amber_energy(ff=ff, nonbonded_method="NoCutoff",
                                        charge_method="FFQ", ensemble=None,
                                        timestep=1e-3, init_temp=1e-3, return_charges=False, ffq_ff=ffq_ff, backprop_solve=False)

  # TODO check log jax compiles on this because it's slow, also look
  # into how to spawn persistent processes instead of running them every time
  nrg_g = jax.jit(jax.value_and_grad(nrg_fn))

  traj_energies = []
  traj_coordinates = []

  dlf_get_params = make_dlf_get_params(coords=pos, printl=0, maxcycle=max_iter)
  dlf_get_gradient = functools.partial(e_g_func, ff=ff, ffq_ff=None, nrg_fn=nrg_g)
  dlf_put_coords = functools.partial(
      store_results, traj_coords=traj_coordinates, traj_energies=traj_energies
  )

  dl_find(
        nvarin=len(pos) * 3,
        dlf_get_gradient=dlf_get_gradient,
        dlf_get_params=dlf_get_params,
        dlf_put_coords=dlf_put_coords,
  )

  energy, grad = nrg_g(traj_coordinates[-1], ff, None)
  return traj_coordinates[-1], traj_energies[-1], grad

DATACLASS_REGISTRY = {"AmberForceField": AmberForceField, "Structure":Structure, "FFQForceField": FFQForceField,
                    "BondRestraint": BondRestraint, "AngleRestraint": AngleRestraint, "TorsionRestraint": TorsionRestraint}

def from_serializable(x):
    if isinstance(x, dict):
        if "__frozendict__" in x:
            return frozendict({k: from_serializable(v) for k, v in x["data"].items()})
        elif "__dataclass__" in x:
            cls = DATACLASS_REGISTRY[x["__dataclass__"]]
            fields = {k: from_serializable(v) for k, v in x["data"].items()}
            return cls(**fields)
        else:
            return {k: from_serializable(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [from_serializable(v) for v in x]
    elif isinstance(x, onp.ndarray):
        return jnp.asarray(x)
    else:
        return x  # primitives

def deserialize_dataclass(filename):
    with open(filename, "rb") as f:
        pytree_loaded = pickle.load(f)
    return from_serializable(pytree_loaded)

def main(coords_file, structs_file, force_fields_file, ffq_ff_file, out_coord_file, out_energy_file, out_grad_file, max_iter):
    # Load and deserialize data structures
    coords = onp.load(coords_file)
    struct = deserialize_dataclass(structs_file)
    force_field = deserialize_dataclass(force_fields_file)
    ffq_ff = deserialize_dataclass(ffq_ff_file)

    # Skip energy minimizaion if structure is not marked for it
    if not struct.energy_minimize:
      nrg_fn, amber_ff, body_fn, state = amber_energy(ff=force_field, nonbonded_method="NoCutoff",
                                        charge_method="FFQ", ensemble=None,
                                        timestep=1e-3, init_temp=1e-3, return_charges=False, ffq_ff=ffq_ff, backprop_solve=False)

      nrg_g = jax.jit(jax.value_and_grad(nrg_fn))

      # TODO it appears the original code may write the energies into the final sum
      # but not the RMSG into the final statistics; this should be considered further
      energy, grad = nrg_g(coords, force_field, None)
    else:
      coords, energy, grad = dlfind_min(coords, struct, force_field, ffq_ff, max_iter)
    
    onp.save(out_coord_file, coords)
    onp.save(out_energy_file, energy)
    onp.save(out_grad_file, grad)

if __name__ == "__main__":
    coords_file, structs_file, force_fields_file, ffq_ff_file, out_coord_file, out_energy_file, out_grad_file, max_iter = sys.argv[1:]
    main(coords_file, structs_file, force_fields_file, ffq_ff_file, out_coord_file, out_energy_file, out_grad_file, int(max_iter))