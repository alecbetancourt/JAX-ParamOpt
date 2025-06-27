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

def dlfind_min(pos, struct, ff, ffq_ff, max_iter, charge_method):
  # TODO is this redundant?
  # dlf_get_params = make_dlf_get_params(coords=pos, maxcycle=max_iter)

  nrg_fn, amber_ff, body_fn, state = amber_energy(ff=ff, nonbonded_method="NoCutoff",
                                        charge_method=charge_method, ensemble=None,
                                        timestep=1e-3, init_temp=1e-3, return_charges=False, ffq_ff=ffq_ff, backprop_solve=False)

  # TODO check log jax compiles on this because it's slow, also look
  # into how to spawn persistent processes instead of running them every time

  NM_TO_BOHR = 18.897259886
  BOHR_TO_NM = 1.0 / 18.897259886
  BOHR_TO_ANGSTROM = 0.52917721092

  def nrg_fn_bohr(x_bohr, ff, nb):
    x_nm = x_bohr / NM_TO_BOHR
    energy, grad_nm = jax.value_and_grad(nrg_fn)(x_nm, ff, nb)
    grad_bohr = grad_nm / NM_TO_BOHR
    return energy, grad_bohr

  nrg_g = jax.jit(nrg_fn_bohr)

  # TODO add debug output
  # if debug
  #   nrg, grad = nrg_g(pos, ff, None, debug=True)
  #   print("debug nrg", nrg)

  # TODO consider if this is completely consistent with all the other code
  # long term the input parsing should handle this, doing this for every
  # energy function invocation isn't particularly clean
  pos = pos/10.0

  pos = pos * NM_TO_BOHR

  traj_energies = []
  traj_coordinates = []

  # TODO just an example
  # TODO consider that these might all be in 1 indexing
  # i think 1 = bond 2 = angle 3 = torsion
  nat = pos.shape[0]
  # condata = jnp.array(
  #   #[3, 2, 1, 9, 8]  # torsion: atoms 1–0–8–7 + 1 for fortran indexing
  #   [[3, 2, 1, 9, 8]]
  #   , dtype=jnp.int32)

  #spec = np.zeros(2*nat + 5*condata.shape[0] + nat, dtype=np.int32) use size instead?
  spec = jnp.zeros(2*nat + 5 + nat, dtype=jnp.int32)
  
  spec = spec.at[:nat].set(1)

  spec = spec.at[nat:2*nat].set(ff.atomic_number)

  # populate atomic fragment + atom types in spec[0:2*nat]
  # ...

  # add constraints starting at offset 2*nat
  j = 2*nat
  for i, row in enumerate(condata):
      #spec[j + 5*i : j + 5*i + 5] = row
      spec = spec.at[j + 5*i : j + 5*i + 5].set(row)

  # should be 1 for hdlc
  spec = spec.at[j + 5:].set(1)

  nspec = 2*nat + 5 + nat
  dlf_get_params = make_dlf_get_params(coords=pos, printl=0, maxcycle=max_iter, maxene=max_iter*7, icoord=1, spec=spec, ncons=1, tatoms=1, nz=nat) # 1 = HDLC
  dlf_get_gradient = functools.partial(e_g_func, ff=ff, ffq_ff=None, nrg_fn=nrg_g)
  dlf_put_coords = functools.partial(
      store_results, traj_coords=traj_coordinates, traj_energies=traj_energies
  )

  dl_find(
        nvarin=len(pos) * 3,
        nspec = nspec,
        dlf_get_gradient=dlf_get_gradient,
        dlf_get_params=dlf_get_params,
        dlf_put_coords=dlf_put_coords,
  )

  energy, grad = nrg_g(traj_coordinates[-1], ff, None)

  # TODO add debug report with number of steps to convergence, time per step
  # ideally breakdown along with final timing breakdown for entire optimization job
  # even in the local/global routines

  # back from nm -> A
  # return traj_coordinates[-1] * 10.0, traj_energies[-1], grad
  return traj_coordinates[-1] * BOHR_TO_ANGSTROM, traj_energies[-1], grad * BOHR_TO_ANGSTROM


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

def main(coords_file, structs_file, force_fields_file, ffq_ff_file, out_coord_file, out_energy_file, out_grad_file, max_iter, charge_method):
    # Load and deserialize data structures
    coords = onp.load(coords_file)
    struct = deserialize_dataclass(structs_file)
    force_field = deserialize_dataclass(force_fields_file)
    ffq_ff = deserialize_dataclass(ffq_ff_file)

    # Skip energy minimizaion if structure is not marked for it
    # TODO add passthrough for pme and other relevant variables
    if not struct.energy_minimize:
      nrg_fn, amber_ff, body_fn, state = amber_energy(ff=force_field, nonbonded_method="NoCutoff",
                                        charge_method=charge_method, ensemble=None,
                                        timestep=1e-3, init_temp=1e-3, return_charges=False, ffq_ff=ffq_ff, backprop_solve=False)

      nrg_g = jax.jit(jax.value_and_grad(nrg_fn))

      # TODO it appears the original code may write the energies into the final sum
      # but not the RMSG into the final statistics; this should be considered further
      energy, grad = nrg_g(coords, force_field, None)
    else:
      coords, energy, grad = dlfind_min(coords, struct, force_field, ffq_ff, max_iter, charge_method)
    
    onp.save(out_coord_file, coords)
    onp.save(out_energy_file, energy)
    onp.save(out_grad_file, grad)

if __name__ == "__main__":
    coords_file, structs_file, force_fields_file, ffq_ff_file, out_coord_file, out_energy_file, out_grad_file, max_iter, charge_method = sys.argv[1:]
    main(coords_file, structs_file, force_fields_file, ffq_ff_file, out_coord_file, out_energy_file, out_grad_file, int(max_iter), charge_method)