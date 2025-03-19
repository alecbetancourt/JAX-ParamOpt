#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of gradient based ReaxFF optimizer using JAX
Includes steepest descent-like energy minimizer (with dynamic LR)
         and scipy-optimizer dependent force field training function

Author: Mehmet Cagri Kaymak
"""
# TODO fix the order and organization of these imports
import numpy as onp
import jax.numpy as jnp
import jax
import time
import copy
from scipy.optimize import minimize
from jax_md.reaxff.reaxff_interactions import calculate_angle
from jax_md.dataclasses import replace, fields
from jax_md.util import safe_mask
from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.reaxff.reaxff_energy import calculate_reaxff_energy
from jaxreaxff.helper import split_dataclass, count_inter_list_sizes, move_dataclass
from jaxreaxff.helper import filter_dataclass, set_params, get_params
from jaxreaxff.interactions import calculate_dist_and_angles, calculate_dist
from frozendict import frozendict
import os
import logging
import math
from jaxreaxff.interactions import DYNAMIC_INTERACTION_KEYS
import time
from jax_md.amber.amber_energy_v2 import amber_energy
from libdlfind import dl_find
from libdlfind.callback import (dlf_get_gradient_wrapper,
                                dlf_put_coords_wrapper, make_dlf_get_params)
import functools
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from flax.serialization import to_bytes, from_bytes
import subprocess
import pickle
from frozendict import frozendict
from jax_md.amber.amber_forcefield import AmberForceField, FFQForceField
from jaxreaxff.structure import Structure, BondRestraint, AngleRestraint, TorsionRestraint

rdndgr = 180.0/onp.pi
dgrrdn = 1.0/rdndgr

def calculate_bond_restraint_energy(positions, structure):
  '''
  Calculate bond restraint potential
  Erestraint= Force1*{1.0-exp(Force2*(distance-target_distance)^2}
  '''
  bond_restraints = structure.bond_restraints
  ind1s = bond_restraints.ind1
  ind2s = bond_restraints.ind2
  targets = bond_restraints.target
  f1s = bond_restraints.force1
  f2s = bond_restraints.force2
  mask = ind1s != -1 # -1 is for the masked values
  #TODO: The distance calculation expects both atoms to be in the center box,
  # does not work when it crosses periodic boundary
  cur_dists = jax.vmap(calculate_dist)(positions[ind1s] - positions[ind2s])
  rest_pot = jnp.sum(mask * f1s *
                     (1.0 - jnp.exp(-f2s * (cur_dists - targets)**2)))
  return rest_pot

def angle_difference(angle1, angle2):
  '''
  Calculate angle difference between 2 angles
  while respecting periodicity
  Ex. diff between 170 and -170 is 20 degree. 
  '''
  diff = jnp.mod(angle1 - angle2, 360)
  diff = jnp.where(diff < 180, diff, 360 - diff)
  return diff

def calculate_angle_restraint_energy(positions, structure):
  '''
  Calculate angle restraint potential
  Erestraint= Force1*{1.0-exp(Force2*(angle-target_angle^2}
  '''

  angle_restraints = structure.angle_restraints

  ind1s = angle_restraints.ind1
  ind2s = angle_restraints.ind2
  ind3s = angle_restraints.ind3
  target = angle_restraints.target
  f1s = angle_restraints.force1
  f2s = angle_restraints.force2
  mask = ind1s != -1
  #TODO: The angle calculation expects both atoms to be in the center box,
  # does not work when it crosses periodic boundary
  disp12 = positions[ind1s] - positions[ind2s]
  disp32 = positions[ind3s] - positions[ind2s]

  # calculate the angle
  cur_angle = jax.vmap(calculate_angle)(disp12, disp32)
  # calculate arccos with safe guards
  cur_angle = safe_mask((cur_angle < 1) & (cur_angle > -1),
                            jnp.arccos, cur_angle).astype(disp12.dtype)
  # convert it to degree
  cur_angle = cur_angle * rdndgr
  # calculate the difference
  diff = angle_difference(cur_angle, target) * dgrrdn
  rest_pot = jnp.sum(mask * f1s * (1.0 - jnp.exp(-f2s * (diff)**2)))

  return rest_pot

def calculate_torsion_angle(p1, p2, p3, p4):
    """
    Calculate the torsion angle
    [1-(2-3)-4] ---- (2-3 is the center)
    Praxeolitic formula
    Taken from: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    """

    b0 = -1.0*(p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= jnp.linalg.norm(b1 + 1e-10)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - jnp.dot(b0, b1)*b1
    w = b2 - jnp.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)
    return jnp.degrees(jnp.arctan2(y, x+1e-10))

def calculate_torsion_restraint_energy(positions, structure):
  '''
  Calculate torsion restraint potential
  Erestraint = Force1*{1.0-exp(Force2*(angle-target_angle^2}
  '''

  torsion_restraints = structure.torsion_restraints

  ind1s = torsion_restraints.ind1
  ind2s = torsion_restraints.ind2
  ind3s = torsion_restraints.ind3
  ind4s = torsion_restraints.ind4
  target = torsion_restraints.target
  f1s = torsion_restraints.force1
  f2s = torsion_restraints.force2
  mask = ind1s != -1
  #TODO: The angle calculation expects both atoms to be in the center box,
  # does not work when it crosses periodic boundary
  cur_angle = jax.vmap(calculate_torsion_angle)(positions[ind1s], positions[ind2s], 
                                                positions[ind3s], positions[ind4s])
  # calculate the difference
  diff = angle_difference(cur_angle, target) * dgrrdn
  rest_pot = jnp.sum(mask * f1s * (1.0 - jnp.exp(-f2s * (diff)**2)))
  return rest_pot


def calculate_energy_and_charges(positions,
                                 structure,
                                 nbr_lists,
                                 force_field,
                                 ff_type,
                                 ffq_ff): # TODO a better way of doing this might be to bundle it in as part of a dict with opt mode and ff type
                                          # also consider using kwargs more like in jax md code for things like this
  '''
  Calculate energy and charges for a given system
  '''

  if ff_type == "reaxff":
    # handle off diag. and symm. in the force field
    force_field = ForceField.fill_off_diag(force_field)
    force_field = ForceField.fill_symm(force_field)

    dists_and_angles = calculate_dist_and_angles(positions,
                                                structure,
                                                nbr_lists)
    # set max_solver_iter to -1 to use direct solve (LU based)
    energy, charges =  calculate_reaxff_energy(structure.atom_types,
                                structure.atomic_nums,
                                nbr_lists,
                                *dists_and_angles,
                                force_field,
                                total_charge = structure.total_charge,
                                tol= 1e-06,
                                backprop_solve = True,
                                tors_2013 = False,
                                solver_model = "EEM",
                                max_solver_iter=-1)
  elif ff_type == "ambereem":
    # TODO placeholder until functions are separated into
    # eventually setup, interactions/distance (returns nrg/nbr), dynamics (returns body fn/state)
    # also separate interactions and nrg that don't require passing functions through nested functions
    # but first
    # setup (returns ff) and separete interactions and energy func, also decouple neighbor list from nrg fn args

    nrg_fn, amber_ff, body_fn, state = amber_energy(ff=force_field, nonbonded_method="NoCutoff",
                                                charge_method="FFQ", ensemble=None,
                                                timestep=1e-3, init_temp=1e-3, return_charges=True, ffq_ff=ffq_ff, backprop_solve=True)

    energy, charges = nrg_fn(positions/10, force_field, None)
  elif ff_type == "amber":
    nrg_fn, amber_ff, body_fn, state = amber_energy(ff=force_field, nonbonded_method="NoCutoff",
                                                charge_method="GAFF", ensemble=None,
                                                timestep=1e-3, init_temp=1e-3, return_charges=True, ffq_ff=ffq_ff, backprop_solve=True)

    energy, charges = nrg_fn(positions/10, force_field, None)

  return energy, charges

def calculate_energy_and_charges_w_rest(positions,
                                 structure,
                                 nbr_lists,
                                 force_field,
                                 ff_type,
                                 ffq_ff):
  '''
  Calculate energy and charges for a given system while inclding the restraints
  '''

  energy, charges = calculate_energy_and_charges(positions,
                                   structure,
                                   nbr_lists,
                                   force_field,
                                   ff_type,
                                   #"amber",
                                   ffq_ff)
  bond_rest_en = calculate_bond_restraint_energy(positions, structure)
  angle_rest_en = calculate_angle_restraint_energy(positions, structure)
  torsion_rest_en = calculate_torsion_restraint_energy(positions, structure)

  energy = energy + bond_rest_en + angle_rest_en + torsion_rest_en
  return energy, charges

def calculate_loss(force_field,
                    list_positions,
                    list_structure,
                    list_nbr_lists,
                    training_data,
                    return_indiv_error=False,
                    ff_type="reaxff",
                    ffq_ff=None):
  '''
  Calculate the loss function
  '''
  # create a dictionary to return individual erros if needed
  all_indiv_errors = dict()
  # Required functions for potential energy and force calculations
  if ff_type == "reaxff":
    pot_f = jax.vmap(calculate_energy_and_charges, in_axes=(0,0,0,None,None,None))
    pot_w_force_f = jax.vmap(jax.value_and_grad(calculate_energy_and_charges,
                                              has_aux=True),
                          in_axes=(0,0,0,None,None,None))
  elif ff_type == "ambereem":
    pot_f = jax.vmap(calculate_energy_and_charges, in_axes=(0,0,0,0,None,None))
    pot_w_force_f = jax.vmap(jax.value_and_grad(calculate_energy_and_charges,
                                              has_aux=True),
                          in_axes=(0,0,0,0,None,None))

  dtype = list_positions[0].dtype

  charge_flag = training_data.charge_items != None
  geo_flag = (training_data.dist_items != None
                    or training_data.angle_items != None
                    or training_data.torsion_items != None)
  force_flag = (training_data.force_items != None
                      or training_data.RMSG_items != None)

  # use onp here to not let get these traced by JAX
  list_sizes = onp.array([len(l.name) for l in list_structure])
  total_num_systems = onp.sum(list_sizes)
  atom_counts = onp.array([l.atom_types.shape[1] for l in list_structure])
  max_atom_count = onp.max(atom_counts)

  total_error = 0.0

  all_energy = jnp.zeros(total_num_systems,dtype=dtype)

  # allocate the data structures to store the results
  if charge_flag:
    all_charges = jnp.zeros((total_num_systems, max_atom_count),dtype=dtype)
  if force_flag:
    all_forces = jnp.zeros((total_num_systems, max_atom_count, 3),dtype=dtype)
  if geo_flag:
    all_positions = jnp.zeros((total_num_systems, max_atom_count, 3),dtype=dtype)

  # evaluate the required observables
  for i in range(len(list_structure)):
    if ff_type == "reaxff":
      list_nbr = list_nbr_lists[i]
      ff = force_field
    elif ff_type == "ambereem":
      list_nbr = []
      ff = force_field[i]
    if force_flag:
      (energy, charges), forces = pot_w_force_f(list_positions[i],
                                                list_structure[i],
                                                list_nbr,
                                                ff,
                                                ff_type,
                                                ffq_ff)
      all_forces = all_forces.at[list_structure[i].name,
                                 :atom_counts[i],
                                 :].set(forces)

    else:
      energy, charges = pot_f(list_positions[i],
                                list_structure[i],
                                list_nbr,
                                ff,
                                ff_type,
                                ffq_ff)
      # TODO also do some runs under a few situations with config.update("jax_debug_nans") or JAX_DEBUG_NANS=True
      # the issue with nan values propogating to gradients from masked indices is still a consideration
      # TODO should add in some kind of test for loss/force gradients at some point
      # if jnp.any(jnp.isnan/isinf/whatever else) -> error

    charges = charges[:, :atom_counts[i]]
    all_energy = all_energy.at[list_structure[i].name].set(energy)
    if charge_flag:
      all_charges = all_charges.at[list_structure[i].name,
                                   :atom_counts[i]].set(charges)
    if geo_flag:
      all_positions = all_positions.at[list_structure[i].name,
                                 :atom_counts[i],
                                 :].set(list_positions[i])


  if training_data.energy_items != None:
    energy_items = training_data.energy_items
    energy_preds = jnp.sum(all_energy[energy_items.sys_inds]
                           * energy_items.multip,axis=1)
    energy_errors = ((energy_items.target - energy_preds) /
                            energy_items.weight) ** 2
    energy_error = jnp.sum(energy_errors)
    total_error += energy_error
    if return_indiv_error:
      all_indiv_errors['ENERGY'] = [energy_preds, energy_items.target, energy_errors]
  if training_data.charge_items != None:
    charge_items = training_data.charge_items
    charge_preds = all_charges[charge_items.sys_ind,charge_items.a_ind]
    charge_errors = ((charge_items.target - charge_preds) /
                           charge_items.weight) ** 2
    charge_error = jnp.sum(charge_errors)
    total_error += charge_error
    if return_indiv_error:
      all_indiv_errors['CHARGE'] = [charge_preds, charge_items.target, charge_errors]
  if training_data.force_items != None:
    force_items = training_data.force_items
    # forces: -1 * grads
    all_forces = all_forces * -1
    force_preds = all_forces[force_items.sys_ind,force_items.a_ind]
    force_errors = ((force_items.target - force_preds) /
                           force_items.weight.reshape(-1,1)) ** 2
    force_error = jnp.sum(force_errors)
    total_error += force_error
    if return_indiv_error:
      all_indiv_errors['FORCE'] = [force_preds, force_items.target, force_errors]

  if training_data.dist_items != None:
    dist_items = training_data.dist_items
    pos1 = all_positions[dist_items.sys_ind,dist_items.a1_ind]
    pos2 = all_positions[dist_items.sys_ind,dist_items.a2_ind]
    disps = pos1 - pos2
    dists = jax.vmap(calculate_dist)(disps)
    dist_errors = ((dist_items.target - dists) /
                           dist_items.weight) ** 2
    dist_error = jnp.sum(dist_errors)
    total_error += dist_error
    if return_indiv_error:
      all_indiv_errors['DISTANCE'] = [dists, dist_items.target, dist_errors]
  if training_data.angle_items != None:
    angle_items = training_data.angle_items
    pos1 = all_positions[angle_items.sys_ind,angle_items.a1_ind]
    pos2 = all_positions[angle_items.sys_ind,angle_items.a2_ind]
    pos3 = all_positions[angle_items.sys_ind,angle_items.a3_ind]
    disp12 = pos1 - pos2
    disp32 = pos3 - pos2
    cur_angle = jax.vmap(calculate_angle)(disp12, disp32)
    # calculate arccos with safe guards
    cur_angle = safe_mask((cur_angle < 1) & (cur_angle > -1),
                              jnp.arccos, cur_angle).astype(disp12.dtype)
    # convert it to degree
    cur_angle = cur_angle * rdndgr
    targets = angle_items.target 
    diff = angle_difference(cur_angle, targets)
    angle_errors = (diff / angle_items.weight) ** 2
    angle_error = jnp.sum(angle_errors)
    total_error += angle_error
    if return_indiv_error:
      all_indiv_errors['ANGLE'] = [cur_angle, targets, angle_errors]

  if training_data.torsion_items != None:
    torsion_items = training_data.torsion_items
    pos1 = all_positions[torsion_items.sys_ind,torsion_items.a1_ind]
    pos2 = all_positions[torsion_items.sys_ind,torsion_items.a2_ind]
    pos3 = all_positions[torsion_items.sys_ind,torsion_items.a3_ind]
    pos4 = all_positions[torsion_items.sys_ind,torsion_items.a4_ind]
    cur_angle = jax.vmap(calculate_torsion_angle)(pos1, pos2, pos3, pos4)
    targets = torsion_items.target 
    diff = angle_difference(cur_angle, targets)
    torsion_errors = (diff / torsion_items.weight) ** 2
    torsion_error = jnp.sum(torsion_errors)
    total_error += torsion_error
    if return_indiv_error:
      all_indiv_errors['TORSION'] = [cur_angle, targets, torsion_errors]
  if return_indiv_error:
    return total_error, all_indiv_errors
  return total_error

@jax.jit
def update_LR(LRs, cur_vals, prev_vals, decr, incr):
  '''
  Update the learning rate based on the energy change in the last iteration
  '''
  LRs = jnp.where(cur_vals < prev_vals, LRs * incr, LRs * decr)
  return LRs

@jax.jit
def update_positions(orig_pos, cur_pos, grads, global_LR, indiv_LRs):
  '''
  Update the positions based on the calculated gradients and learning rate
  '''
  # clip the gradients to not allow them to explode
  # adhoc evaluations show that the provided clipping works well in practice
  clip = 10.0
  grads = jnp.clip(grads, -clip, clip)
  grads = jnp.where(jnp.abs(grads) < 0.1, 0.0, grads)
  grads = grads * (global_LR * indiv_LRs).reshape(-1,1,1)

  # do not allow position to change more than 0.01 Angstrom in one step
  change_vec = jnp.clip(grads, -0.01, +0.01)
  new_pos = cur_pos - change_vec

  diff = orig_pos - new_pos
  # do not allow positions to move more than 1 A from the initial values
  diff = jnp.clip(diff, -1.0, 1.0)
  new_pos = orig_pos - diff
  return new_pos

def update_inter_sizes(positions, structures, force_field,
                       cur_sizes, multip=1.5):
  '''
  Update the interaction list sizes if the current sizes are not enough to
  hold all of the interactions
  '''
  structures = replace(structures, positions=positions)
  force_field = move_dataclass(force_field, onp)
  structures = move_dataclass(structures, onp)
  list_structures = split_dataclass(structures)
  num_threads = os.cpu_count()
  sizes = count_inter_list_sizes(list_structures, force_field,
                                 num_threads=num_threads, chunksize=4)
  
  for k in DYNAMIC_INTERACTION_KEYS:
    for s in sizes:
        # assign some buffer room
        s[k] = math.ceil(s[k] * multip)
  # pick the maximum size for each interaction
  max_sizes = copy.deepcopy(dict(cur_sizes))
  for k in max_sizes.keys():
    for s in sizes:
      max_sizes[k] = max(max_sizes[k], s[k])
  return frozendict(max_sizes)

@jax.jit
def calculate_RMSG(grads, atom_counts):
  '''
  Calculate root mean squared gradients (forces)
  '''
  return jnp.sqrt((jnp.sum(grads**2, axis=(1,2)) + 1e-5) /
                  (atom_counts.reshape(-1) * 3))


def subprocess_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Completely hide GPUs in subprocess
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Force JAX CPU backend explicitly
    os.environ["JAX_PLATFORMS"] = "cpu"      # Enforce JAX CPU-only backend

    # Import JAX only after setting these env vars
    global jax, jnp
    import jax
    import jax.numpy as jnp

### Serialization helpers - partially adapted from flax.serialization

# Map of dataclass name to actual class for reconstruction
DATACLASS_REGISTRY = {"AmberForceField": AmberForceField, "Structure":Structure, "FFQForceField": FFQForceField,
              "BondRestraint": BondRestraint, "AngleRestraint": AngleRestraint, "TorsionRestraint": TorsionRestraint}

def to_serializable(x):
  if isinstance(x, jax.Array):
      return onp.array(x)  # safely converts to numpy
  elif isinstance(x, frozendict):
      return {"__frozendict__": True, "data": {k: to_serializable(v) for k, v in x.items()}}
  elif isinstance(x, dict):
      return {k: to_serializable(v) for k, v in x.items()}
  elif isinstance(x, (list, tuple)):
      return [to_serializable(v) for v in x]
  elif hasattr(x, '__dataclass_fields__'):
      return {
          "__dataclass__": type(x).__name__,
          "data": {k: to_serializable(getattr(x, k)) for k in x.__dataclass_fields__}
      }
  else:
      return x  # primitives

def serialize_dataclass(dc, filename):
  pytree_serializable = to_serializable(dc)
  with open(filename, "wb") as f:
      pickle.dump(pytree_serializable, f)

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
  elif isinstance(x, np.ndarray):
    return jnp.asarray(x)
  else:
    return x  # primitives

def deserialize_dataclass(filename):
  with open(filename, "rb") as f:
    pytree_loaded = pickle.load(f)
  return from_serializable(pytree_loaded)

def serialize_numpy(arr, filename):
  onp.save(filename, arr)

def deserialize_numpy(filename):
  return onp.load(filename)

def run_subprocess(coords, structs, force_fields, ffq_ff, minim_steps, idx):
  coords_file = f"./tmp/coords_{idx}.npy"
  structs_file = f"./tmp/structs_{idx}.pkl"
  force_fields_file = f"./tmp/force_fields_{idx}.pkl"
  ffq_ff_file = f"./tmp/ffq_ff_{idx}.pkl"
  out_coord_file = f"./tmp/out_coord_{idx}.npy"
  out_energy_file = f"./tmp/out_energy_{idx}.npy"
  out_grad_file = f"./tmp/out_grad_{idx}.npy"

  # Serialize inputs
  serialize_numpy(coords, coords_file)
  serialize_dataclass(structs, structs_file)
  serialize_dataclass(force_fields, force_fields_file)
  serialize_dataclass(ffq_ff, ffq_ff_file)

  # Launch subprocess explicitly
  script_path = os.path.abspath(__file__)
  script_directory = os.path.dirname(script_path)
  python_executable = sys.executable

  subprocess.run([
      python_executable, script_directory + "/dlfind.py",
      coords_file, structs_file, force_fields_file, ffq_ff_file, out_coord_file, out_energy_file, out_grad_file, str(minim_steps)
  ], stdout=subprocess.DEVNULL)

  # Load result
  out_coord = onp.load(out_coord_file)
  out_energy = onp.load(out_energy_file)
  out_grad = onp.load(out_grad_file)

  # Clean up files
  os.remove(coords_file)
  os.remove(structs_file)
  os.remove(force_fields_file)
  os.remove(ffq_ff_file)
  os.remove(out_coord_file)
  os.remove(out_energy_file)
  os.remove(out_grad_file)

  return out_coord, out_energy, out_grad

def dlfind_opt(list_structure,
                center_sizes, force_field, ffq_ff,
                allocate_func, force_func, ff_type,
                init_LR = 0.001,
                minim_steps = 100,
                target_RMSG = 1.0):
  '''
  Perform geometry optimization using callback to DL FIND library

  Args:

  Returns:

  '''
  sub_filter = [data.energy_minimize for data in list_structure]
  list_sub_structure = [filter_dataclass(data, sub_filter[i])
                            for i, data in enumerate(list_structure)]

  list_sub_structure = list_structure
  
  dtype = list_structure[0].positions.dtype
  list_pos = [s.positions for s in list_structure]
  list_sub_cur_pos = [s.positions for s in list_sub_structure]

  cur_center_sizes = list(center_sizes)
  cur_loss_vals = [0] * len(center_sizes)

  full_RMSG_vals = [jnp.zeros_like(l.energy_minimize, dtype=dtype)
                  for l in list_structure]

  for i in range(len(center_sizes)):
    batch_size = list_sub_cur_pos[i].shape[0]

    coords_list = [jnp.array(list_sub_cur_pos[i][j]) for j in range(batch_size)] # TODO onp for these three?
    struct_list = [
      #move_dataclass(jax.tree_util.tree_map(lambda x: x[j], list_sub_structure[i]), onp)
      jax.tree_util.tree_map(lambda x: x[j], list_sub_structure[i])
      for j in range(batch_size)
    ]
    ff_list = [
      #move_dataclass(jax.tree_util.tree_map(lambda x: x[j], force_field[i]), onp)
      jax.tree_util.tree_map(lambda x: x[j], force_field[i])
      for j in range(batch_size)
    ]

    optimized_coords_list = []
    num_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
      futures = [
        executor.submit(run_subprocess, coords_list[i], struct_list[i], ff_list[i], ffq_ff, minim_steps, i)
        for i in range(batch_size)
      ]
      results = [future.result() for future in futures]

    traj_coordinates, traj_energies, final_grad = zip(*results)
    traj_coordinates = jnp.stack(traj_coordinates) # TODO onp or jnp? had as onp originally
    traj_energies = jnp.stack(traj_energies)
    final_grad = jnp.stack(final_grad)

    # TODO add option to print individual RMSG and energies
    # particularly important when some structures aren't optimized
    # they will have disproportionately high energy/RMSG
    # print("RMSG", jax.jit(calculate_RMSG)(final_grad, list_sub_structure[i].atom_count))

    list_sub_cur_pos[i] = traj_coordinates
    cur_loss_vals[i] = jnp.sum(traj_energies)
    cur_RMSG = jax.jit(calculate_RMSG)(final_grad, list_sub_structure[i].atom_count)
    full_RMSG_vals[i] = full_RMSG_vals[i].at[list_structure[i].energy_minimize].set(cur_RMSG)

  cur_total_loss = sum(cur_loss_vals)
  for i in range(len(list_pos)):
    if len(list_sub_structure[i].energy_minimize) > 0:
      list_pos[i] = list_pos[i].at[list_structure[i].energy_minimize].set(list_sub_cur_pos[i])
  return list_pos, cur_total_loss, cur_center_sizes, full_RMSG_vals


def energy_minimize(list_structure,
                    center_sizes, force_field, ffq_ff,
                    allocate_func, force_func, ff_type,
                    init_LR = 0.001,
                    minim_steps = 100,
                    target_RMSG = 1.0):
  # DL-FIND option
  if ff_type == "ambereem":
    [list_pos, cur_total_loss, 
    cur_center_sizes, full_RMSG_vals] = dlfind_opt(list_structure,
                    center_sizes, force_field, ffq_ff,
                    allocate_func, force_func, ff_type,
                    init_LR, minim_steps, target_RMSG)
    return list_pos, cur_total_loss, cur_center_sizes, full_RMSG_vals
  
  '''
  Energy minimize the given structures
  '''
  # select the structures that require energy minimization
  list_sub_structure = [filter_dataclass(data, data.energy_minimize)
                             for data in list_structure]
  dtype = list_structure[0].positions.dtype # TODO replace this with something else
  # create place holder for LR
  LRs = [jnp.ones_like(l.energy_minimize, dtype=dtype)
              for l in list_sub_structure]
  minim_steps_list = [l.energy_minim_steps
              for l in list_sub_structure]
  global_LR = jnp.float32(init_LR)
  # handtuned increment and decrement amounts to update learning rate
  # LR increases if there is a continous decrease in energy
  # and decrease the LR if the energy increases
  decr = jnp.float32(0.75)
  incr = jnp.float32(1.1)
  list_prev_energy = [None] * len(center_sizes)

  list_pos = [s.positions for s in list_structure]
  list_sub_cur_pos = [s.positions for s in list_sub_structure]
  list_sub_orig_pos = copy.deepcopy(list_sub_cur_pos)
  cur_center_sizes = list(center_sizes)
  cur_loss_vals = [0] * len(center_sizes)
  # placeholder to store RMSG values
  full_RMSG_vals = [jnp.zeros_like(l.energy_minimize, dtype=dtype)
                   for l in list_structure]
  sub_RMSG_vals = [None] * len(center_sizes)

  for iter_c in range(minim_steps):
    for i in range(len(center_sizes)):
      if len(list_sub_structure[i].energy_minimize) > 0 and jnp.any(LRs[i] > 0):
        if ff_type == "reaxff":
          sub_nbr, counts = allocate_func(list_sub_cur_pos[i],
                                          list_sub_structure[i],
                                          force_field, cur_center_sizes[i])
          if jnp.any(sub_nbr.did_buffer_overflow):
            print(f"Interaction list overflow for cluster-{i+1} during energy minimization at step {iter_c + 1}!")
            new_cluster_center = update_inter_sizes(list_sub_cur_pos[i],
                                                    list_sub_structure[i],
                                                    force_field,
                                                    cur_center_sizes[i],
                                                    multip=1.5)
            print("name: old size -> new size")
            for k in counts.keys():
              #if cur_center_sizes[i][k] != new_cluster_center[k]:
              print(f"{k}: {cur_center_sizes[i][k]}->{new_cluster_center[k]}")           
            cur_center_sizes[i] = new_cluster_center
            # repopulate the neighbor list since the other one is invalidated
            sub_nbr = allocate_func(list_sub_cur_pos[i],
                                            list_sub_structure[i],
                                            force_field, cur_center_sizes[i])[0]
        elif ff_type == "ambereem":
          sub_nbr = []
          ff = force_field[i]
          
        (energy, ch), grads = force_func(list_sub_cur_pos[i],
                                 list_sub_structure[i],
                                 sub_nbr,
                                 ff,
                                 ff_type,
                                 ffq_ff)

        cur_loss_vals[i] = jnp.sum(energy)
        cur_RMSG = jax.jit(calculate_RMSG)(grads, list_sub_structure[i].atom_count)
        sub_RMSG_vals[i] = cur_RMSG
        full_RMSG_vals[i] = full_RMSG_vals[i].at[list_structure[i].energy_minimize].set(cur_RMSG)

        if iter_c > 0:
            LRs[i] = jax.jit(update_LR)(LRs[i], energy, list_prev_energy[i], decr, incr)


        LRs[i] = LRs[i] * (minim_steps_list[i] > 0) * (sub_RMSG_vals[i] > target_RMSG)
        minim_steps_list[i] = minim_steps_list[i] - 1
        list_sub_cur_pos[i] = jax.jit(update_positions)(list_sub_orig_pos[i],
                                                        list_sub_cur_pos[i],
                                                        grads,
                                                        global_LR,
                                                        LRs[i])
        list_prev_energy[i] = energy
    cur_total_loss = sum(cur_loss_vals)
  # update the positions
  for i in range(len(list_pos)):
    if len(list_sub_structure[i].energy_minimize) > 0:
      list_pos[i] = list_pos[i].at[list_structure[i].energy_minimize].set(list_sub_cur_pos[i])

  return list_pos, cur_total_loss, cur_center_sizes, full_RMSG_vals

def add_noise_to_params(params, bounds, scale=0.001):
  '''
  Add noise to the parameters if the optimizer is stuck
  '''
  # produce noise from [-1,1]
  noise = onp.random.uniform(low = -1.0, high = 1.0,size=len(params))
  new_params = params + (noise * scale) * (bounds[:,1] - bounds[:,0])
  new_params = jnp.clip(new_params, bounds[:,0], bounds[:,1])

  return new_params

def lower_bounds(params, bounds, advanced_opts):
  '''
  Lower the parameter boundaries to restrict the search further
  to reduce error fluctuations
  '''

  new_bounds = []
  for j, b in enumerate(bounds):
    lower_bound, upper_bound = b
    size = (upper_bound - lower_bound) * advanced_opts['perc_width_rest_search']
    param = float(params[j])
    lower_bound = max(lower_bound, param - size)
    upper_bound = min(upper_bound, param + size)
    lower_bound = min(lower_bound, upper_bound)
    upper_bound = max(lower_bound, upper_bound)
    new_bounds.append((lower_bound, upper_bound))
  new_bounds = onp.array(new_bounds, dtype=bounds.dtype)
  return new_bounds

def random_parameter_search(bounds, sample_count,
                            param_indices, force_field, training_data,
                            list_positions, aligned_data, center_sizes,
                            loss_func):

  args = (param_indices, force_field, training_data,
          list_positions, aligned_data, center_sizes)
  dtype = force_field.gamma.dtype
  min_loss = float('inf')
  min_params = None
  for _ in range(sample_count):
    selected_params = onp.random.uniform(low=bounds[:,0],high=bounds[:,1])
    selected_params = jnp.array(selected_params, dtype=dtype)
    loss = loss_func(selected_params, *args)
    if loss < min_loss or onp.isnan(min_loss) == True:
      min_loss = loss
      min_params = selected_params
  if min_params != None and onp.isnan(min_loss) == False:
    selected_params = min_params
    print("Loss after random search (w/o energy minim.): ", min_loss)
  else:
    selected_params = onp.random.uniform(low=bounds[:,0],high=bounds[:,1])
    selected_params = jnp.array(selected_params, dtype=dtype)
  return selected_params



def train_FF(params, param_indices, param_bounds, force_field,
             list_structure, center_sizes, training_data,
             validation_data,
             iter_count, e_minim_flag, optimizer, optim_options,
             advanced_opts,
             loss_and_grad_func, minim_func, allocate_func, ff_type, opt_mode, ffq_ff):
  '''
  Main parameter optimization routine
  '''

  prev_true_loss = float('inf')
  global_min = float('inf')
  global_min_params = jnp.array(copy.deepcopy(params))
  get_params_jit = jax.jit(get_params,static_argnums=(1,2,3))
  set_params_jit = jax.jit(set_params,static_argnums=(1,3,4))
  total_f_ev = 0
  list_positions = [s.positions for s in list_structure]
  param_lower_flag = False
  true_current_loss = 0.0
  surrogate_loss = 0.0
  for e in range(iter_count+1):
    print("*" * 40)
    print("Iteration: {}".format(e))
    iteration_start = time.time()
    # assign params to ff
    if (e > 1
        and (iter_count - e) == 5
        and abs(surrogate_loss - global_min) / global_min > 0.2
        and param_lower_flag == False):
      print("Restricting to search further reduce loss fluctuations from energy minimization")
      param_bounds = lower_bounds(global_min_params, param_bounds, advanced_opts)
      params = global_min_params
      param_lower_flag = True

    if ff_type == "reaxff":
      force_field = set_params_jit(force_field, param_indices, params, ff_type, opt_mode)
    elif ff_type == "ambereem":
      ffq_ff = set_params_jit(ffq_ff, param_indices, params, ff_type, opt_mode)
    if e_minim_flag:
      minim_start = time.time()
      [list_positions, cur_total_energy,
      center_sizes, cur_RMSG_vals] = minim_func(list_structure,
                                                center_sizes,
                                                force_field, ffq_ff)
      minim_end = time.time()
      print("Energy minimization took {:.4f} sec.".format(minim_end-minim_start))
      print('  Total pot. E: {:.2f} kcal/mol'.format(cur_total_energy))
      # print informative RMSG info to show how many structures are properly
      # energy minimized
      for rmsg_limit in [1.0, 2.5, 5.0]:
        count = sum([onp.sum((RMSG.reshape(-1)>rmsg_limit)
                             * s.energy_minim_steps.reshape(-1) > 50)
                     for RMSG, s in zip(cur_RMSG_vals,list_structure)])
        print('  RMSG > {:.1} count:{}'.format(rmsg_limit,count))
    elif ff_type == "reaxff": # TODO change when allocation is implemented for amber
      # extend the interaction list sizes if needed
      for i in range(len(list_structure)):
        sub_nbr, new_c = allocate_func(list_positions[i], list_structure[i],
                                   force_field, center_sizes[i])
        if jnp.any(sub_nbr.did_buffer_overflow):
          print(f"Interaction list overflow for cluster-{i+1} during training!")
          new_cluster_center = update_inter_sizes(list_positions[i],
                                                   list_structure[i],
                                                   force_field,
                                                   center_sizes[i],
                                                   multip=1.5)
          
          print("name: old size -> new size")
          for k in new_c.keys():
            #if center_sizes[i][k] != new_cluster_center[k]:
            print(f"{k}: {center_sizes[i][k]}->{new_cluster_center[k]}")
          center_sizes[i] = new_cluster_center

    # calculate the true loss, right after energy minimization
    prev_true_loss = true_current_loss
    true_train_loss, _ = loss_and_grad_func(params, param_indices,
                                              force_field, training_data,
                                              list_positions, list_structure,
                                              center_sizes, ff_type, opt_mode, ffq_ff)
    true_train_loss = float(true_train_loss)
    # calculate the validation loss
    # if valid. data is available, total loss = valid loss + train loss
    if validation_data != None:
      true_valid_loss, _ = loss_and_grad_func(params, param_indices,
                                            force_field, validation_data,
                                            list_positions, list_structure,
                                            center_sizes, ff_type, opt_mode, ffq_ff)
      true_valid_loss = float(true_valid_loss)
      print("True training loss: {:.2f}".format(true_train_loss))
      print("True validation loss: {:.2f}".format(true_valid_loss))
      true_current_loss = true_valid_loss + true_train_loss
    else:
      true_current_loss =  true_train_loss 
    true_current_loss = float(true_current_loss)
    print("True total loss: {:.2f}".format(true_current_loss))
    # save the parameters if best
    if true_current_loss < global_min:
      global_min = true_current_loss
      global_min_params = jnp.array(copy.deepcopy(params))
      print("Lowest true total loss so far: {:.2f}".format(global_min))
    # add noise if stuck
    if (abs(true_current_loss - prev_true_loss)
        / max(true_current_loss, prev_true_loss) < advanced_opts['perc_err_change_thr']):
      params = add_noise_to_params(global_min_params, param_bounds,
                                   scale=advanced_opts['perc_noise_when_stuck'])
      print('noise is added to the best parameter set so far')
      continue
    # error minimization, unless we are at the last step
    if e < iter_count:
      args = (param_indices,
              force_field, training_data,
              list_positions, list_structure, center_sizes, ff_type, opt_mode, ffq_ff)
      min_state = minimize(loss_and_grad_func, params, jac=True, args=args,
                           method=optimizer,
                           bounds=param_bounds,options=optim_options)
      print("funv.ev {}, surrogate loss after loss minim.: {:.2f}".format(min_state.nfev,
                                                                      min_state.fun))
      surrogate_loss = min_state.fun
      total_f_ev = total_f_ev + min_state.nfev
      params = jnp.array(min_state.x)
    iteration_end = time.time()
    print("Iteration took {:.4f}".format(iteration_end-iteration_start))
  return global_min_params, global_min, center_sizes

