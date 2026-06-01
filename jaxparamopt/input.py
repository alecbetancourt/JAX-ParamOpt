"""Backend-aware input loading interfaces for jaxparamopt."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
import h5py
import jax
import jax.numpy as jnp
import numpy as onp
from jax_md import dataclasses
from typing import Any, Mapping
from .backends import get_backend_input_loader
from .config import OptimizationConfig
from jaxparamopt.trainingdata import ChargeItem, EnergyItem, DistItem, AngleItem 
from jaxparamopt.trainingdata import TorsionItem, ForceItem, RMSGItem, HessianItem, TrainingData
from jaxparamopt.structure import Structure, BondRestraint, AngleRestraint, TorsionRestraint


@dataclass(slots=True)
class ParameterInput:
  """Logical parameter-space input produced by the loading stage."""

  raw_parameters: Any = None
  parameter_ids: tuple[Any, ...] = ()
  bounds: tuple[tuple[float, float], ...] = ()
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InputBundle:
  """Logical handoff between configuration and downstream workflow stages."""

  config: OptimizationConfig
  backend_name: str
  raw_model: Any = None
  raw_systems: Any = None
  training_data: Any = None
  validation_data: Any = None
  parameter_input: ParameterInput = field(default_factory=ParameterInput)
  metadata: dict[str, Any] = field(default_factory=dict)


def load_inputs(
    config: OptimizationConfig | Mapping[str, object],
) -> InputBundle:
  """Load backend-aware optimization inputs from a normalized configuration."""
  if isinstance(config, OptimizationConfig):
    normalized_config = config
  else:
    normalized_config = OptimizationConfig.from_mapping(config)

  loader = get_backend_input_loader(normalized_config.ff_type)
  return loader.load_inputs(normalized_config)


# ===========================================================================
# GEOMETRY LOADING
# ===========================================================================

def orthogonalization_matrix(box_lengths, angles_degr):
  '''
  Calculate a transformation matrix to be used in distance calculations
  with periodic boundary condition
  '''
  # to calculate shifted box coord: mat.dot(shift_array)
  # source: reaxff frotran, subroutine vlist
  a,b,c = box_lengths
  angles = onp.radians(angles_degr)
  sina, sinb, sing = onp.sin(angles)
  cosa, cosb, cosg = onp.cos(angles)
  cosphi = (cosg - cosa * cosb)/(sina * sinb)
  if cosphi >  1.0:
    cosphi = 1.0
  sinphi = onp.sqrt(1.0 - cosphi*cosphi)
  #tm11,tm21,tm31,tm22,tm32,tm33
  mat =  onp.array((
          (a * sinb * sinphi,       0.0,                0.0),
          (a * sinb * cosphi,      b * sina,           0.0),
          (a *  cosb,              b * cosa,            c)),
          dtype=jnp.float32)

  if angles_degr[0] == 90.0 and angles_degr[1] == 90.0 and angles_degr[2] == 90.0:
    mat = onp.eye(3, dtype=jnp.float32)
    mat[0,0] = a
    mat[1,1] = b
    mat[2,2] = c
  return mat

def calculate_box_shifts(is_periodic, far_nbr_cutoff, orth_mat):
  '''
  Find all different boxes we can see given the cutoff
  '''
  if is_periodic == True:
    kx_limit = math.ceil(far_nbr_cutoff / orth_mat[0,0])
    ky_limit = math.ceil(far_nbr_cutoff / orth_mat[1,1])
    kz_limit = math.ceil(far_nbr_cutoff / orth_mat[2,2])

  else:
    kx_limit = 0
    ky_limit = 0
    kz_limit = 0
  kx_list = list(range(-kx_limit, kx_limit+1))
  ky_list = list(range(-ky_limit, ky_limit+1))
  kz_list = list(range(-kz_limit, kz_limit+1))

  all_shift_comb = [[0,0,0]] # this should be the first one
  for x in kx_list:
    for y in ky_list:
      for z in kz_list:
        # this is already added
        if x == 0 and y == 0 and z == 0:
          continue
        all_shift_comb.append((x,y,z))
  return onp.array(all_shift_comb)

def read_geo_file(geo_file, name_to_index_map, ff_type, far_nbr_cutoff=10.0):
  '''
  Read the geometries from the provided geometry file
  '''
  if not os.path.exists(geo_file):
    print("Path {} does not exist!".format(geo_file))
    return []

  if os.path.splitext(geo_file)[-1] in ['.h5', '.hdf5']:
    print("[INFO] Input geometry format is detected as .h5")
    list_systems = []
    with h5py.File(geo_file, 'r') as hf:
      # TODO is this necessary, or is there a more automated way of making
      # h5py and jax more interoperable?
      # TODO clean this up and merge the two code paths after these details are worked out
      # TODO consider if doing this two step group approach is really the best way
      # it could also help to have some main /structures/cluster... group
      # and then figure out how to grab all the leaves of /structures
      # more flexible going forward too as you don't iterate over every group explicitly
      for key in hf:
        if key == "training":
            continue
        group = hf[key]
        num_structures = group.attrs["structure_count"]
        system_name = key
        num_atoms = group.attrs["nat"]
        atom_names = onp.char.decode(group["species"][()], encoding='utf-8')
        reax_atom_types = onp.array([name_to_index_map[name] for name in atom_names])
        atomic_nums = onp.zeros(num_atoms, dtype=onp.int32)
        atomic_nums = onp.where(atom_names=="C", 6, atomic_nums)
        atomic_nums = onp.where(atom_names=="O", 8, atomic_nums)
        atomic_nums = onp.where(atom_names=="H", 1, atomic_nums)
        atoms_positions = group["coordinates"][()]
        box = [999.0, 999.0, 999.0]
        box_angles = [90.0,90.0,90.0]
        is_periodic = False
        do_minimization = True
        max_it = 99999
        # box information
        orth_mat = orthogonalization_matrix(box, box_angles)
        all_shifts = calculate_box_shifts(is_periodic, far_nbr_cutoff, orth_mat)
        # TODO add restraints
        bond_restraints = [[-1,-1,0,0,0]]
        angle_restraints = [[-1,-1,-1,0,0,0]]
        torsion_restraints = [[-1,-1,-1,-1,0,0,0]]
        bond_restraints = onp.array(bond_restraints)
        angle_restraints = onp.array(angle_restraints)
        torsion_restraints = onp.array(torsion_restraints)
        total_charge = onp.zeros((num_structures,)) # TODO add non 0 charge
        new_bond_restraints = BondRestraint(ind1 = bond_restraints[:,0].astype(onp.int32),
                                            ind2 = bond_restraints[:,1].astype(onp.int32),
                                            force1 = bond_restraints[:,2].astype(onp.float32),
                                            force2 = bond_restraints[:,3].astype(onp.float32),
                                            target = bond_restraints[:,4].astype(onp.float32))

        new_angle_restraints = AngleRestraint(ind1 = angle_restraints[:,0].astype(onp.int32),
                                            ind2 = angle_restraints[:,1].astype(onp.int32),
                                            ind3 = angle_restraints[:,2].astype(onp.int32),
                                            force1 = angle_restraints[:,3].astype(onp.float32),
                                            force2 = angle_restraints[:,4].astype(onp.float32),
                                            target = angle_restraints[:,5].astype(onp.float32))

        new_torsion_restraints = TorsionRestraint(ind1 = torsion_restraints[:,0].astype(onp.int32),
                                            ind2 = torsion_restraints[:,1].astype(onp.int32),
                                            ind3 = torsion_restraints[:,2].astype(onp.int32),
                                            ind4 = torsion_restraints[:,3].astype(onp.int32),
                                            force1 = torsion_restraints[:,4].astype(onp.float32),
                                            force2 = torsion_restraints[:,5].astype(onp.float32),
                                            target = torsion_restraints[:,6].astype(onp.float32))
        # TODO this should be changed
        # training items should be separate from the structures in principle
        # the question is how to best organize and iterate over them
        # it looks like these values may also be dummy values anyways?
        # where are they used?
        # TODO maybe remove from structures.py?
        target_e = 0.0
        # if "/training/energy_items" in hf:
        #   target_e = group["energies"][()]
        
        target_f = onp.zeros((num_atoms, 3), dtype=onp.float32) # TODO add force, charge, hess
        # if "/training/force_items" in hf:
        #   target_f = group["forces"][()]
        
        target_ch = onp.zeros((num_atoms), dtype=onp.float32)
        # if "/training/charge_items" in hf:
        #   target_ch = group["charges"][()]

        # TODO this might not scale well for empty storage
        # maybe use none as a default value for these
        # target_hess = onp.zeros((num_structures, num_atoms, num_atoms), dtype=onp.float32)

        for i in range(num_structures):
          new_system = Structure(system_name + f"_{i}", num_atoms,
                                  reax_atom_types, atomic_nums,
                                  atoms_positions[i], orth_mat, total_charge[i],
                                  do_minimization, max_it, all_shifts,
                                  new_bond_restraints, new_angle_restraints,
                                  new_torsion_restraints,
                                  target_e,target_f,target_ch) # target values are not used here

          list_systems.append(new_system)

    return list_systems

  list_systems = []
  f = open(geo_file,'r')
  system_name = ''
  atoms_positions = []
  # add dummy restraints
  bond_restraints = [[-1,-1,0,0,0]]
  angle_restraints = [[-1,-1,-1,0,0,0]]
  torsion_restraints = [[-1,-1,-1,-1,0,0,0]]
  molcharge_items = []
  atom_names = []
  system_str = ''
  box = [999.0, 999.0, 999.0]
  box_angles = [90.0,90.0,90.0]
  is_periodic = False
  do_minimization = True
  max_it = 99999
  for line in f:
    if len(line.strip()) > 2:
      system_str = system_str + line
    line = line.split('#', 1)[0]
    if line.strip().startswith('#') or len(line) < 1:
      continue
    if line.startswith('END'):
      num_atoms = len(atom_names)
      # currently only total charge for all of the atom is supported, no partial charges
      if (len(molcharge_items) > 1 
          or (len(molcharge_items) == 1 
              and (molcharge_items[0][1] - molcharge_items[0][0] + 1) < num_atoms)):
        print("[ERROR] error in {}, MOLCHARGE is only supported for the total system charge!".format(system_name))
        sys.exit()

      total_charge = 0
      if len(molcharge_items) == 1:
        total_charge = molcharge_items[0][2]
      atom_names = onp.array(atom_names)
      atomic_nums = onp.zeros(num_atoms, dtype=onp.int32)
      # The rest of the atomic numbers are not important
      # TODO recording atomic numbers for AMBER probably isn't necessary internally, but double check
      if ff_type == "reaxff":
        atomic_nums = onp.where(atom_names=="C", 6, atomic_nums)
        atomic_nums = onp.where(atom_names=="O", 8, atomic_nums)
        atomic_nums = onp.where(atom_names=="H", 1, atomic_nums)
        reax_atom_types = [name_to_index_map[name] for name in atom_names]
        reax_atom_types = onp.array(reax_atom_types)
      elif ff_type == "ambereem" or ff_type == "amber":
        #TODO just change this to atom types
        # if the map is a list, then each of the structures corresponds to the same index in the map list
        # so it needs to be something like reax_atom_types = [name_to_index_map[str_idx][name] for name in atom_names]
        if isinstance(name_to_index_map, list):
          str_idx = len(list_systems)
          reax_atom_types = [name_to_index_map[str_idx][name] for name in atom_names]
        else:
          reax_atom_types = [name_to_index_map[name] for name in atom_names]
        reax_atom_types = onp.array(reax_atom_types)
      atoms_positions = onp.array(atoms_positions)
      # box information
      orth_mat = orthogonalization_matrix(box, box_angles)
      all_shifts = calculate_box_shifts(is_periodic, far_nbr_cutoff, orth_mat)
      # restraints
      bond_restraints = onp.array(bond_restraints)
      angle_restraints = onp.array(angle_restraints)
      torsion_restraints = onp.array(torsion_restraints)

      new_bond_restraints = BondRestraint(ind1 = bond_restraints[:,0].astype(onp.int32),
                                          ind2 = bond_restraints[:,1].astype(onp.int32),
                                          force1 = bond_restraints[:,2].astype(onp.float32),
                                          force2 = bond_restraints[:,3].astype(onp.float32),
                                          target = bond_restraints[:,4].astype(onp.float32))

      new_angle_restraints = AngleRestraint(ind1 = angle_restraints[:,0].astype(onp.int32),
                                          ind2 = angle_restraints[:,1].astype(onp.int32),
                                          ind3 = angle_restraints[:,2].astype(onp.int32),
                                          force1 = angle_restraints[:,3].astype(onp.float32),
                                          force2 = angle_restraints[:,4].astype(onp.float32),
                                          target = angle_restraints[:,5].astype(onp.float32))

      new_torsion_restraints = TorsionRestraint(ind1 = torsion_restraints[:,0].astype(onp.int32),
                                          ind2 = torsion_restraints[:,1].astype(onp.int32),
                                          ind3 = torsion_restraints[:,2].astype(onp.int32),
                                          ind4 = torsion_restraints[:,3].astype(onp.int32),
                                          force1 = torsion_restraints[:,4].astype(onp.float32),
                                          force2 = torsion_restraints[:,5].astype(onp.float32),
                                          target = torsion_restraints[:,6].astype(onp.float32))
      # filler values
      target_e = 0
      target_f = onp.zeros_like(atoms_positions, dtype=onp.float32)
      target_ch = onp.zeros_like(reax_atom_types, dtype=onp.float32)
      # create the structure from the read data
      new_system = Structure(system_name, num_atoms,
                             reax_atom_types, atomic_nums,
                             atoms_positions, orth_mat, total_charge,
                             do_minimization, max_it, all_shifts,
                             new_bond_restraints, new_angle_restraints,
                             new_torsion_restraints,
                             target_e,target_f,target_ch) # target values are not used here

      list_systems.append(new_system)
      atoms_positions = []
      atom_names = []
      # add dummy restraints
      bond_restraints = [[-1,-1,0,0,0]]
      angle_restraints = [[-1,-1,-1,0,0,0]]
      torsion_restraints = [[-1,-1,-1,-1,0,0,0]]
      molcharge_items = []
      system_str = ''
      box = [999.0, 999.0, 999.0]
      box_angles = [90.0,90.0,90.0]
      is_periodic = False
      do_minimization = True
      max_it = 99999
    else:
      if line.startswith('DESCRP'):
        system_name = line.strip().split()[1]
      # box info
      elif line.startswith('CRYSTX'):
        line = line.strip().split()
        x = float(line[1])
        y = float(line[2])
        z = float(line[3])
        x_ang = float(line[4])
        y_ang = float(line[5])
        z_ang = float(line[6])
        box = [x,y,z]
        box_angles = [x_ang,y_ang,z_ang]
        is_periodic = True
      # whether we need energy minim.
      elif line.startswith('RUTYPE'):
        if line.find('SINGLE') > -1:
          do_minimization = False
          max_it = 0 # means full minimization

        elif line.find('NORMAL RUN') > -1:
          do_minimization = True
          max_it = 99999 # means full minimization
        #for now assume all of them are the same
        #TODO: fix this
        elif line.find('MAXIT') > -1:
          max_it = int(line.strip().split()[-1])

          if max_it < 5:
            max_it = 0
            do_minimization = False
      # charge info
      elif line.startswith('MOLCHARGE'):
        #Ex. MOLCHARGE   1  30  1.00
        split_line = line.split()[1:]
        at1, at2 = split_line[:2]
        total_charge = float(split_line[2])
        molcharge_items.append([int(at1)-1,int(at2)-1,total_charge])
      # restraint info
      elif line.startswith('BOND RESTRAINT'):
        split_line = line.split()[2:]
        at1,at2 = split_line[:2]
        dist = split_line[2]
        force1,force2 = split_line[3:5]
        d_dist = split_line[5]
        bond_restraints.append([int(at1)-1,int(at2)-1,float(force1),float(force2),float(dist)])
      elif line.startswith('ANGLE RESTRAINT'):
        split_line = line.split()[2:]
        at1,at2,at3 = split_line[:3]
        angle = split_line[3]
        force1,force2 = split_line[4:6]
        d_angle = split_line[6]
        angle_restraints.append([int(at1)-1,int(at2)-1,float(at3)-1,float(force1),float(force2),float(angle)])
      elif line.startswith('TORSION RESTRAINT'):
        split_line = line.split()[2:]
        at1,at2,at3,at4 = split_line[:4]
        torsion = split_line[4]
        force1,force2 = split_line[5:7]
        d_torsion = split_line[7]
        torsion_restraints.append([int(at1)-1,int(at2)-1,int(at3)-1,int(at4)-1,float(force1),float(force2),float(torsion)])
      elif line.startswith('HETATM'):
        line = line.strip().split()
        atom_index = int(line[1])
        atom_name = line[2]
        x = float(line[3])
        y = float(line[4])
        z = float(line[5])
        atom_pos = [x,y,z]
        atoms_positions.append(atom_pos)
        atom_names.append(atom_name)

  f.close()

  return list_systems

# ===========================================================================
# TRAINING SET LOADING
# ===========================================================================

def preprocess_trainset_line(line):
  '''
  Proeprocess a given line from training set
  '''
  line = line.replace('/', ' / ')
  return line

def read_train_set(train_in):
  '''
  Read the training set data
  '''
  if os.path.splitext(train_in)[-1] in ['.h5', '.hdf5']:
    # TODO need to add support for multiple items per nrg training item
    # TODO also need to finish adding other training items for hdf parsing
    # should probably look at disulfide/cobalt example
    # h5 variable length strings?
    training_items = {}
    energy_items = []
    force_items = []
    hessian_items = []
    with h5py.File(train_in, 'r') as hf:
      # energy items
      if "training/energy_items" in hf:
        for i in range(len(hf["/training/energy_items/energies"])):
          name_list = [name.decode(encoding='utf-8') for name in hf["/training/energy_items/names"][i]]
          multiplier_list = list(hf["/training/energy_items/multipliers"][i])
          energy = hf["/training/energy_items/energies"][i]
          weight = hf["/training/energy_items/weights"][i]
          energy_item = EnergyItem(name_list, multiplier_list, energy, weight)
          energy_items.append(energy_item)
      
      # force items
      # TODO think about loading everything in one go and then iterating
      # it seems like this may be somewhat slow
      # or figure out how to do vmapped instantiation of training items
      if "training/force_items" in hf:
        start_time = time.time()
        for i in range(len(hf["/training/force_items/forces"])):
          name_list = hf["/training/force_items/names"][i].decode(encoding='utf-8')
          weight = hf["/training/force_items/weights"][i]
          indices = hf["/training/force_items/indices"][i]
          forces = hf["/training/force_items/forces"][i]

          force_item = ForceItem(name_list, indices, [forces[0],forces[1],forces[2]], weight)
          force_items.append(force_item)
        end_time = time.time()
        print("Force item loading time:", end_time-start_time)

      # hessian items
      if "training/hessian_items" in hf:
        start_time = time.time()
        for i in range(len(hf["/training/hessian_items/names"])):
          name_list = hf["/training/hessian_items/names"][i].decode(encoding='utf-8')
          weight = hf["/training/hessian_items/weights"][i]
          indices = hf["/training/hessian_items/indices"][i]

          # TODO figure out better way of doing this
          # flattening the hessian isn't exactly trivial
          # in this case, will be NxN,3,3 in shape versus n,3,n,3 shape from jax.hessian
          hessian = hf["/training/hessian_items/hessians"][i]

          hessian_item = HessianItem(name_list, indices, hessian, weight)
          hessian_items.append(hessian_item)
        end_time = time.time()
        print("Hessian item loading time:", end_time-start_time)

    training_items['energy_items'] = energy_items
    training_items['force_items'] = force_items
    training_items['hessian_items'] = hessian_items
    return training_items

  f = open(train_in, 'r')
  training_items = {}
  energy_flag = 0
  charge_flag = 0
  geo_flag = 0
  force_flag = 0
  new_RMSG_flag = 0 # use to minimize forces
  geo2_items = []
  geo3_items = []
  geo4_items = []
  force_RMSG_items = []
  force_atom_items = []
  energy_items = []
  charge_items = []

  for line in f:
    #print(line)
    line = line.strip()
    # ignore everything after #
    line = line.split('#', 1)[0]
    line = line.split('!', 1)[0]
    if len(line) == 0 or line.startswith("#"):
      continue
    # flags to use to detect corresponding regions
    elif line.startswith("ENERGY"):
      energy_flag = 1

    elif line.startswith("CHARGE"):
      charge_flag = 1

    elif line.startswith("GEOMETRY"):
      geo_flag = 1

    elif line.startswith('FORCES'):
      force_flag = 1

    elif line.startswith("ENDENERGY"):
      energy_flag = 0

    elif line.startswith("ENDCHARGE"):
      charge_flag = 0

    elif line.startswith("ENDGEOMETRY"):
      geo_flag = 0

    elif line.startswith("ENDFORCES"):
      force_flag = 0
    # energy items
    elif energy_flag == 1:
      line = preprocess_trainset_line(line)
      split_line = line.split()
      # w and energy + 4 items per ref. item
      num_ref_items = int((len(split_line) - 2) / 4) 

      name_list = []
      multiplier_list = []
      weight = float(split_line[0])
      for i in range(num_ref_items):
          div = float(split_line[4 * i + 4].strip())
          mult = 1/div
          if split_line[1 + 4*i].strip() == '+':
              multiplier_list.append(mult)
          else:
              multiplier_list.append(-mult)


          name_list.append(split_line[4 * i + 2].strip())

      energy = float(split_line[-1])
      energy_item = EnergyItem(name_list, multiplier_list, energy, weight)

      energy_items.append(energy_item)
    # charge item
    elif charge_flag == 1:
      line = preprocess_trainset_line(line)
      split_line = line.split()
      name = split_line[0].strip()
      weight = float(split_line[1])
      index = int(split_line[2]) - 1
      charge = float(split_line[3])
      charge_item = ChargeItem(name, index, charge, weight)
      charge_items.append(charge_item)
    # geo item
    elif geo_flag == 1:
      line = preprocess_trainset_line(line)
      split_line = line.split()
      name = split_line[0].strip()
      weight = float(split_line[1])
      target = float(split_line[-1])
      # 2-body
      if len(split_line) == 5:
        index1 = int(split_line[2]) - 1
        index2 = int(split_line[3]) - 1
        dist_item = DistItem(name, index1, index2, target, weight)
        geo2_items.append(dist_item)

      # 3-body
      if len(split_line) == 6:
        index1 = int(split_line[2]) - 1
        index2 = int(split_line[3]) - 1
        index3 = int(split_line[4]) - 1
        angle_item = AngleItem(name, index1, index2, index3, target, weight)
        geo3_items.append(angle_item)
      # 4-body
      if len(split_line) == 7:
        index1 = int(split_line[2]) - 1
        index2 = int(split_line[3]) - 1
        index3 = int(split_line[4]) - 1
        index4 = int(split_line[5]) - 1
        torsion_item = TorsionItem(name, index1, index2, index3, index4, target, weight)
        geo4_items.append(torsion_item)
      #RMSG
      if len(split_line) == 3:
        rmsg_item = RMSGItem(name, target, weight)
        force_RMSG_items.append(rmsg_item)
    # force item
    elif force_flag == 1:
      split_line = line.split()
      line = preprocess_trainset_line(line)
      split_line = line.split()
      name = split_line[0].strip()
      weight = float(split_line[1])
      #force on indiv. atoms
      if len(split_line) == 6:
        index = int(split_line[2]) - 1
        f1 = float(split_line[3])
        f2 = float(split_line[4])
        f3 = float(split_line[5])
        force_item = ForceItem(name, index, [f1,f2,f3], weight)
        force_atom_items.append(force_item)

  if len(energy_items) > 0:
    training_items['energy_items'] = energy_items

  if len(charge_items) > 0:
    training_items["charge_items"] = charge_items

  if len(geo2_items) > 0:
    training_items["dist_items"] = geo2_items

  if len(geo3_items) > 0:
    training_items["angle_items"] = geo3_items

  if len(geo4_items) > 0:
    training_items["torsion_items"] = geo4_items

  if len(force_RMSG_items) > 0:
    training_items["RMSG_items"] = force_RMSG_items

  if len(force_atom_items) > 0:
    training_items["force_items"] = force_atom_items

  return training_items

# ===========================================================================
# PARAMETER SET LOADING
# ===========================================================================

def read_parameter_file(params_file, ignore_sensitivity=1):
  '''
  Read the parameter file
  '''
  # section indices sensitivity low_end high_end !comments
  if not os.path.exists(params_file):
    return
  params = []
  f = open(params_file,'r')

  group_flag = False
  # TODO the control flow for this function is kind of messy
  # in the case of multiple force field files, the first index is the ff index
  for line in f:
    # remove comments
    line = line.split('!')[0]
    line = line.split('#')[0]
    split_line = line.strip().split()
    if split_line[0] == "GROUP":
      group_items = []
      group_flag = True
      sensitivity = float(split_line[1])
      low_end = float(split_line[2])
      high_end = float(split_line[3])
      if ignore_sensitivity:
        sensitivity = 1
      if low_end > high_end:
        temp = low_end
        low_end = high_end
        high_end = temp
      continue
    elif split_line[0] == "ENDGROUP":
      group_flag = False
      params.append((group_items, sensitivity, low_end, high_end))
      continue
    if group_flag == True:
      # TODO can also just determine this based on line length
      #if mode == "single":
      if len(split_line) == 3:
        section = int(split_line[0])
        index1 = int(split_line[1])
        index2 = int(split_line[2])
        item = (section,index1,index2)
      #elif mode == "multi":
      elif len(split_line) == 4:
        ff_index = int(split_line[0])
        section = int(split_line[1])
        index1 = int(split_line[2])
        index2 = int(split_line[3])
        item = (ff_index,section,index1,index2)
      else:
        raise ValueError("Error in reading group item from parameter file, should either be 3 or 4 columns")
      group_items.append(item)
      continue
      # have to figure out how to group all of the items for a group
    # TODO better error handling is needed
    if len(split_line) < 6:
      continue
    #if mode == "single":
    if len(split_line) == 6:
      section = int(split_line[0])
      index1 = int(split_line[1])
      index2 = int(split_line[2])
      sensitivity = float(split_line[3])
      low_end = float(split_line[4])
      high_end = float(split_line[5])
      if ignore_sensitivity:
        sensitivity = 1
      if low_end > high_end:
        temp = low_end
        low_end = high_end
        high_end = temp
      item = (section,index1,index2,sensitivity, low_end, high_end)
    #elif mode == "multi":
    elif len(split_line) == 7:
      ff_index = int(split_line[0])
      section = int(split_line[1])
      index1 = int(split_line[2])
      index2 = int(split_line[3])
      sensitivity = float(split_line[4])
      low_end = float(split_line[5])
      high_end = float(split_line[6])
      if ignore_sensitivity:
        sensitivity = 1
      if low_end > high_end:
        temp = low_end
        low_end = high_end
        high_end = temp
      item = (ff_index,section,index1,index2,sensitivity, low_end, high_end)
    else:
      raise ValueError("Error in reading item from parameter file, should either be 6 or 7 columns")
    params.append(item)
  return params

def map_params(params, index_map):
  '''
  Map the read parameters to new type of indexing to select them from
  a given force field object
  '''
  # TODO this can probably be removed and folded into the cluster target generation
  # index map can either be for a single ff object
  # or can be a list of ff objects where .params_to_indices
  # is the map for each ff in the list
  # in that case, the format will be p[0:3] where p[0] is the ff index in the list
  # and p[1],p[2],p[3] are the indicies into the map for that ff

  # params can either be
  # (ff_index,section,index1,index2,sensitivity, low_end, high_end)
  # (section,index1,index2,sensitivity, low_end, high_end)
  # (group_items, sensitivity, low_end, high_end)
  
  new_params = []
  
  for p in params:
    if len(p) not in [4,6,7]:
      raise ValueError("Error in parameter format, should either be 4,6 or 7 items per line")
    if len(p) == 4: # group item
      group_item = []
      for item in p[0]:
        ff_index,section,index1,index2 = item
        idx_map = index_map[ff_index].params_to_indices
        key = (section,index1,index2)
        value = idx_map[key]
        group_item.append((value,ff_index))
      # lists aren't hashable
      new_item = (("group", tuple(group_item)), p[1],p[2],p[3])
      new_params.append(new_item)
    elif len(p) == 6: # single ff item
      if isinstance(index_map, list):
        raise ValueError("Error in parameter mapping, cannot have single ff item with multiple ff index map")
      key = (p[0],p[1],p[2])
      value = ("single", index_map[key])
      new_item = (value, p[3],p[4],p[5])
      new_params.append(new_item)
    elif len(p) == 7: # multi ff item
      if not isinstance(index_map, list):
        raise ValueError("Error in parameter mapping, cannot have multi ff item with single ff index map")
      idx_map = index_map[p[0]].params_to_indices
      key = (p[1],p[2],p[3])
      value = ("multi", p[0], idx_map[key])
      new_item = (value, p[4],p[5],p[6])
      new_params.append(new_item)

  return new_params

# ===========================================================================
# GENERAL INPUT UTILITIES
# ===========================================================================

def create_structure_map(structures):
  '''
  Create name -> index and index->name maps
  '''
  name_to_index = {}
  index_to_name = {}
  for i in range(len(structures)):
    s = structures[i]
    name_to_index[s.name] = i
    index_to_name[i] = s.name
  return name_to_index, index_to_name

def filter_data(systems, training_items):
  '''
  Filter out the unused items
  '''
  system_names = {s.name for s in systems}
  new_systems = []
  new_training_items = {}
  used_geo_names = set()
  for key in training_items.keys():
    new_training_items[key] = []
    for item in training_items[key]:
      if key == 'energy_items':
        names = item.sys_inds
      else:
        names = [item.sys_ind,]
      skip = False
      for name in names:
        if name not in system_names:
          skip = True
          break
      if skip == False:
        new_training_items[key].append(item)
        for name in names:
          used_geo_names.add(name)
  new_systems = [s for s in systems if s.name in used_geo_names]
  return new_systems, new_training_items

def structure_training_data(training_items, geo_name_to_index):
  '''
  Restructure the training data items to be used for training
  '''
  # replace names with indices
  for key in training_items.keys():
      for i, item in enumerate(training_items[key]):
          if key == 'energy_items':
              sys_inds = [geo_name_to_index[name] for name in item.sys_inds]
              item = dataclasses.replace(item, sys_inds = sys_inds)
          else:
              sys_ind = geo_name_to_index[item.sys_ind]
              item = dataclasses.replace(item, sys_ind = sys_ind)
          training_items[key][i] = item
  # Align the sizes for the energy items
  if 'energy_items' in training_items:
      energy_items = training_items['energy_items']
      max_sys_len = max([len(item.sys_inds) for item in energy_items])
      for i, item in enumerate(energy_items):
          sys_inds = item.sys_inds
          filler = [-1] * (max_sys_len - len(sys_inds))
          sys_inds.extend(filler)

          multip = item.multip
          filler = [0.0] * (max_sys_len - len(multip))
          multip.extend(filler)

          energy_items[i] = dataclasses.replace(energy_items[i],
                                                multip=multip,
                                                sys_inds=sys_inds)
      training_items['energy_items'] = energy_items

  new_items = {}
  for key in training_items.keys():
      items = training_items[key]
      if len(items) == 0:
          continue
      field_names = [field.name for field in dataclasses.fields(items[0])]
      collected_attr = {}
      for attr in field_names:
          attr_list = []
          for item in items:
              val = getattr(item, attr)
              attr_list.append(val)
          collected_attr[attr] = onp.array(attr_list)
      collected_obj = items[0].__class__(**collected_attr)
      new_items[key] = collected_obj
  return TrainingData(**new_items)
