#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver code to run the trainer

Author: Mehmet Cagri Kaymak
"""
import os, sys
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
#TODO: Figure out how to eliminate oom issus
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
#TODO: Implement type switching between single and double precision
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_disable_jit", True)
import jax.profiler
import jax_md
import jax.numpy as jnp
import numpy as onp
import time
import argparse
from jaxreaxff.smartformatter import SmartFormatter
from frozendict import frozendict
from jax_md.reaxff.reaxff_energy import calculate_reaxff_energy
from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.reaxff.reaxff_helper import read_force_field
from jax_md import dataclasses
from jaxreaxff.optimizer_v2 import (calculate_loss,
                                 calculate_energy_and_charges_w_rest,
                                 add_noise_to_params, random_parameter_search,
                                 train_FF, energy_minimize, update_inter_sizes)
from jaxreaxff.helper import set_params, get_params, produce_error_report, count_inter_list_sizes
from jaxreaxff.interactions import (reaxff_interaction_list_generator,
                                    calculate_dist_and_angles,
                                    DYNAMIC_INTERACTION_KEYS)
from jaxreaxff.structure import align_structures, align_and_batch_structures
from jaxreaxff.helper import (move_dataclass, process_and_cluster_geos,
                              create_structure_map, read_parameter_file,
                              map_params, read_geo_file, read_train_set,
                              filter_data, structure_training_data,
                              parse_and_save_force_field)
import math
from functools import partial

def main():
  # create parser for command-line arguments
  parser = argparse.ArgumentParser(description='JAX-ReaxFF driver',
                                   formatter_class=SmartFormatter)
  # default inputs: inital force field, parameters, geo and trainset files
  parser.add_argument('--init_FF', metavar='filename',
      type=str,
      default="ffield",
      help='Initial force field file')
  parser.add_argument('--params', metavar='filename',
      type=str,
      default="params",
      help='Parameters file')
  parser.add_argument('--geo', metavar='filename',
      type=str,
      default="geo",
      help='Geometry file')
  parser.add_argument('--train_file', metavar='filename',
      type=str,
      default="trainset.in",
      help='Training set file')
  parser.add_argument('--use_valid', metavar='boolean',
      type=bool,
      default=False,
      help='Flag indicating whether to use validation data (True/False)')
  parser.add_argument('--valid_file', metavar='filename',
      type=str,
      default="validset.in",
      help='Validation set file (same format as trainset.in)')
  parser.add_argument('--valid_geo_file', metavar='filename',
      type=str,
      default="valid_geo",
      help='Geo file for the validation data')
  # optimization related parameters
  parser.add_argument('--opt_method', metavar='method',
      choices=['L-BFGS-B', 'SLSQP'],
      type=str,
      default='L-BFGS-B',
      help='Optimization method - "L-BFGS-B" or "SLSQP"')
  parser.add_argument('--num_trials', metavar='number',
      type=int,
      default=1,
      help='R|Number of trials (Population size).\n' +
      'If set to <= 0, provided force field will be evaluated w/o any training (init_FF).')
  parser.add_argument('--num_steps', metavar='number',
      type=int,
      default=5,
      help='Number of optimization steps per trial')
  parser.add_argument('--init_FF_type', metavar='init_type',
      choices=['random', 'educated', 'fixed'],
      default='fixed',
      help='R|How to start the trials from the given initial force field.\n' +
      '"random": Sample the parameters from uniform distribution between given ranges.\n'
      '"educated": Sample the parameters from a narrow uniform distribution centered at given values.\n'
      '"fixed": Start from the parameters given in "init_FF" file')
  parser.add_argument('--random_sample_count', metavar='number',
      type=int,
      default=0,
      help='R|Before the optimization starts, uniforms sample the paramater space.\n' +
      'Select the best sample to start the training with, only works with "random" inital start.\n' +
      'if set to 0, no random search step will be skipped. ')
  # energy minimization related parameters
  parser.add_argument('--num_e_minim_steps', metavar='number',
      type=int,
      default=0,
      help='Number of energy minimization steps')
  parser.add_argument('--e_minim_LR', metavar='init_LR',
      type=float,
      default=5e-4,
      help='Initial learning rate for energy minimization')
  parser.add_argument('--end_RMSG', metavar='end_RMSG',
      type=float,
      default=1.0,
      help='Stopping condition for E. minimization')
  # output related options
  parser.add_argument('--out_folder', metavar='folder',
      type=str,
      default="outputs",
      help='Folder to store the output files')
  parser.add_argument('--save_opt', metavar='option',
      choices=['all', 'best'],
      default="best",
      help='R|"all" or "best"\n' +
      '"all": save all of the trained force fields\n' +
      '"best": save only the best force field')
  parser.add_argument('--cutoff2', metavar='cutoff',
      type=float,
      default=0.001,
      help='BO-cutoff for valency angles and torsion angles')
  parser.add_argument('--max_num_clusters', metavar='max # clusters',
      type=int,
      default=10,
      choices=range(1, 16),
      help='R|Max number of clusters that can be used\n' +
           'High number of clusters lowers the memory cost\n' +
           'However, it increases compilation time,especially for cpus')
  parser.add_argument('--seed', metavar='seed',
      type=int,
      default=0,
      help='Seed value')
  parser.add_argument('--ff_type', metavar='ff_type',
      choices=['reaxff', 'amber', 'ambereem'],
      type=str,
      default='reaxff',
      help='Forcefield to optimize - "reaxff" or "amber" or "ambereem"')
  parser.add_argument('--ref_ene', metavar='filename',
      type=str,
      default="ref_ene.json",
      help='Reference Energy file for AMBER Optimization')
  parser.add_argument('--torsions', metavar='filename',
      type=str,
      default="torsions.json",
      help='Torsions file for AMBER Optimization')
  parser.add_argument('--init_FF_amber', metavar='filename',
      type=str,
      default="Datasets/amber/xyztoprmtop/3_create_geo_multiple_atomtypes_gaff",
      help='Directory where AMBER force field files will be generated for geometries')
  parser.add_argument('--generate_prmtop', metavar='value',
      type=bool,
      default=False,
      help='Boolean to determine if prmtop files need to be generated from .xyz files')

  #parse arguments
  args = parser.parse_args()

  ff_type_map = {'reaxff':0, 'amber':1, 'ambereem':2}
  ff_type_int = ff_type_map[args.ff_type]
  # TODO: Rationalize default values and types to enable seamless switching with missing arguments
  aligned_amber_ff = None

  #TODO: fix imports and remove anything that isn't necessary
  import subprocess
  if(args.generate_prmtop):
    # should error stream be seperated?
    #foo > stdout.txt 2> stderr.txt
    #foo > allout.txt 2>&1
    #TODO: add error handling here if anything appears in stderr
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/../Datasets/amber/xyztoprmtop'
    with open('convert.stdout', 'w') as stdout_f, open('convert.stderr', 'w') as stderr_f:
      subprocess.call("./convert.sh", cwd=script_path, stdout = stdout_f, stderr = stderr_f)

  #sys.exit()

  from jaxreaxff.optimizer_amber import ff_opt
  if(args.ff_type == 'amber'):
    #run amber optimizer externally and pass ff args
    #(prmtop_dir, params_dir, geo_dir, min_steps, opt_loops, opt_interval, ref_ene, torsions)
    # ./jaxreaxff/driverv3.py --init_FF ./Datasets/amber/prmtop                            \
    #           --params ./Datasets/amber/dh_6-7-9-11/params.json                          \
    #           --geo ./Datasets/amber/dh_6-7-9-11/confs_999-999/dh_6-7-9-11/dh_6-7-9-11_  \
    #           --num_e_minim_steps 2000                                                   \
    #           --num_trials 1                                                             \
    #           --num_steps 5                                                              \
    #           --ref_ene ./Datasets/amber/ref_ene.json                                    \
    #           --torsions ./Datasets/amber/torsions.json                                  \
    #           --ff_type amber
    print("[INFO] Starting AMBER Optimizer")
    print("[WARNING] AMBER Optimization without EEM not currently supported")
    #ff_opt(args.init_FF, args.params, args.geo, args.num_e_minim_steps, args.num_trials, args.num_steps, args.ref_ene, args.torsions)
    #sys.exit(0)

  import jax_md.amber.amber_energy as amber
  from jaxreaxff.generate_prmtop import build_prm_list
  from jaxreaxff.structure_amber import load_ff, align_forcefield, align_and_batch_forcefield
  import openmm as omm
  import openmm.app as app
  # if(args.ff_type == 'ambereem'):
  #   print("[INFO] Starting AMBER Optimization w/ EEM")
  #   # TODO: this eventually needs to be rolled into a dataclass and some work will have to be done to create a treemap
  #   # for vmap compatibility, see: https://stackoverflow.com/questions/73765064/jax-vmap-over-batch-of-dataclasses
  #   # the aligning done to the list structures may also serve this purpose if there isn't an underlying mechanism that works
  #   flist = build_prm_list(args.geo, args.init_FF_amber)
  #   #print("Flist", flist)
  #   prm_dict_list, max_sizes = load_ff(flist)
  #   # for prm in prm_dict_list:
  #   #   for key, value in prm.items():
  #   #     print(key, value)
  #   # sys.exit()
    
  #   #aligned_amber_ff = align_forcefield(prm_dict_list, max_sizes)
  #   ff_and_geo_batch_size = 40

  #   aligned_amber_ff = align_and_batch_forcefield(prm_dict_list, max_sizes, ff_and_geo_batch_size)

  #   aligned_amber_ff = [move_dataclass(d, jnp) for d in aligned_amber_ff]
  #   print("aligned ff len", len(aligned_amber_ff))
  #   print("type", type(aligned_amber_ff[0]))

    #print(aligned_amber_ff)
    #sys.exit()

    # prmlist = []
    # #print(flist)
    # for f in flist:
    #   prmtop = omm.app.AmberPrmtopFile(f)
    #   bondprm = amber.bond_init(prmtop._prmtop)
    #   angleprm = amber.angle_init(prmtop._prmtop)
    #   torsionprm = amber.torsion_init(prmtop._prmtop)
    #   ljprm = amber.lj_init(prmtop._prmtop)
    #   coulprm = amber.coul_init(prmtop._prmtop)
    #   prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)
    #   prmlist.append(prms)
    # # TODO: create function to convert orthoginalization matrix back to vectors or convert structure to store this information
    # #print(prmlist)
    # print("Amber Params Length Setup", len(prmlist))
    # prmidx = jnp.arange(len(prmlist))
    # boxVectors = jnp.array([999.9,999.9,999.9])
    
    

  # TODO: remove
  args.save_opt = "all"
  default_backend = jax.default_backend().lower()

  if default_backend == 'cpu':
    print("[WARNING] Falling back to CPU")
    print("To use the GPU version, jaxlib with CUDA support needs to installed!")

  # advanced options
  advanced_opts = {"perc_err_change_thr":0.01,       # if change in error is less than this threshold, add noise
                   "perc_noise_when_stuck":0.04,     # noise percantage (wrt param range) to add when stuck
                   "perc_width_rest_search":0.15,    # width of the restricted parameter search after iteration > rest_search_start
                   }

  onp.random.seed(args.seed)
  TYPE = jnp.float64
  # read the initial force field
  force_field = read_force_field(args.init_FF, cutoff2 = args.cutoff2, dtype=TYPE)
  force_field = ForceField.fill_off_diag(force_field)
  force_field = ForceField.fill_symm(force_field)

  # print INFO
  print("[INFO] Force field field is read")
  ###########################################################################
  #read the paramemters to be optimized
  params_list_orig = read_parameter_file(args.params, ignore_sensitivity=0)
  params_list = map_params(params_list_orig, force_field.params_to_indices)

  # preprocess params
  param_indices=[]
  for par in params_list:
      param_indices.append(par[0])
  param_indices = tuple(param_indices)

  bounds = []
  for p in params_list:
      bounds.append((p[2],p[3]))
  bounds = onp.array(bounds)
  # print INFO
  print(f"[INFO] Parameter file is read, there are {len(param_indices)} parameters to be optimized!")
  ###########################################################################


  # read the geo file
  systems = read_geo_file(args.geo, force_field.name_to_index, 10.0)

  print("[INFO] Number of geometries loaded:", len(systems))

  training_data = read_train_set(args.train_file)
  # default value for the valid. data
  validation_data = None
  systems_tr, training_data = filter_data(systems, training_data)
  # read and process the validation data if used
  if args.use_valid:
    print("[INFO] Validation data is provided!")
    systems_valid = read_geo_file(args.valid_geo_file, force_field.name_to_index, 10.0)
    validation_data = read_train_set(args.valid_file)
    systems_valid, validation_data = filter_data(systems_valid, validation_data)
    # combine training and validation data together (geo files)
    used_geo_names = set([s.name for s in systems_tr])
    systems = systems_tr
    for sys in systems_valid:
      if sys.name not in used_geo_names:
        systems.append(sys)
  else:
     systems = systems_tr

  geo_name_to_index, geo_index_to_name = create_structure_map(systems)
  training_data = structure_training_data(training_data, geo_name_to_index)
  if args.use_valid:
     validation_data = structure_training_data(validation_data, geo_name_to_index)
  # replace names with indices
  for i,s in enumerate(systems):
      s = dataclasses.replace(s, name = geo_name_to_index[s.name])
      systems[i] = s
  


  #print("system 485 positions", systems[484].positions)
  #print("len system 485 positions", len(systems[484].positions))
  #sys.exit




  if(args.ff_type == 'ambereem'):
      print("[INFO] Starting AMBER Optimization w/ EEM")

      ff_and_geo_batch_size = 40

      num_threads = os.cpu_count()
      #num_threads = 1
      #num_threads = 32
      #print("num_threads", num_threads)
      #update this to something like
      #https://stackoverflow.com/questions/76308447/how-to-use-properly-slurm-sbatch-and-python-multiprocessing
      #ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
      #should hopefully return the actual allocated number of cpus

      # [globally_sorted_indices,
      # all_cut_indices,
      # center_sizes] = process_and_cluster_geos(systems, force_field,
      #                                           max_num_clusters=args.max_num_clusters,
      #                                           num_threads=num_threads,
      #                                           chunksize=4,
      #                                           close_cutoff=5.0, far_cutoff=10.0)

      # print("Process and cluster done")

      #batched alloc scheme
      #find max periodic image counts and atom counts
      #cluster geos by 20
      #count interaction lists for each cluster in loop
      #ideally only count far nbr sizes or figure out alterantive far nbr scheme
      #would jax md neighbor lists work with indices passed to eem?
      #center sizes = each cluster's size

      flist = build_prm_list(args.geo, args.init_FF_amber)
      #print("FLIST", flist)

      def cluster_structs(structures, batch_size, dtype=onp.float64):
        full_size = len(structures)
        # max_sizes = {'num_atoms':0, 'periodic_image_count':0}
        # for struct in structures:
        #   max_sizes['num_atoms'] = max(max_sizes['num_atoms'], len(struct.atom_types))
        #   max_sizes['periodic_image_count'] = len(struct.periodic_image_shifts)
        batches = []
        ffields = []
        center_sizes = []
        for bs in range(0,full_size,batch_size):
          # max_sizes = {'num_atoms': 0,
          #      'periodic_image_count': 0,
          #      'far_nbr_size': 300,
          #      'close_nbr_size': 300,
          #      'filter2_size': 300,
          #      'filter3_size': 300,
          #      'filter4_size': 300,
          #      'hbond_size': 300,
          #      'hbond_h_size': 300,
          #      'hbond_filter_far_size': 300,
          #      'hbond_filter_close_size': 300}
          max_sizes = {'num_atoms': 0,
               'periodic_image_count': 0,
               'far_nbr_size': 300}
          for struct in structures[bs:bs+batch_size]:
            atom_mask = struct.atom_types != -1
            max_sizes['num_atoms'] = max(max_sizes['num_atoms'], len(atom_mask))
            max_sizes['periodic_image_count'] = max(max_sizes['periodic_image_count'], len(struct.periodic_image_shifts))
          batch = align_structures(structures[bs:bs+batch_size],
                              max_sizes, dtype)
          batches.append(batch)
          center_sizes.append(max_sizes)
          #print("flists", flist[bs:bs+batch_size])
          # prm_dict_list, ff_sizes = load_ff(flist[bs:bs+batch_size])
          # ffield = align_forcefield(prm_dict_list, ff_sizes, dtype)
          # ffields.append(ffield)
          
          # batch_sizes = count_inter_list_sizes(structures[bs:bs+batch_size], force_field, 
          #                               num_threads=num_threads, chunksize=4,
          #                               close_cutoff=5.0,
          #                               far_cutoff=10.0)
          # center_size = batch_sizes[0]
          # for size in batch_sizes:
          #   for k in center_size.keys():
          #     center_size[k] = max(center_size[k], size[k])
          # center_sizes.append(center_size)


        return batches, center_sizes
        # return batches, ffields, center_sizes

      aligned_data, center_sizes = cluster_structs(systems, ff_and_geo_batch_size)
      #aligned_data, aligned_amber_ff, center_sizes = cluster_structs(systems, ff_and_geo_batch_size)
      print("[INFO] Clustering Finished, number of clusters:", len(center_sizes))
      aligned_data = [move_dataclass(d, jnp) for d in aligned_data]
      # print("center sizes", center_sizes)
      # print("center size len", len(center_sizes))
      #print("center -1 len", len(center_sizes[-1]))

      #TODO: remove openmm dependency and improve performance of prmtop loader
      aligned_amber_ff = []
      for bs in range(0,len(systems),ff_and_geo_batch_size):
        prm_dict_list, ff_sizes = load_ff(flist[bs:bs+ff_and_geo_batch_size])
        ffield = align_forcefield(prm_dict_list, ff_sizes, onp.float64)
        aligned_amber_ff.append(ffield)

      #atom_types = structure.atom_types
      #num_atoms = len(atom_types)
      #size_dict["num_atoms"] = len(atom_mask)
      #size_dict["periodic_image_count"] = len(structure.periodic_image_shifts)




      # size_dicts = count_inter_list_sizes(systems, force_field, 
      #                                   num_threads=num_threads, chunksize=4,
      #                                   close_cutoff=5.0,
      #                                   far_cutoff=10.0)
      #print("Inter list size counting done")
      #print("center sizes", center_sizes)

      #sys.exit()


      #print("size_dicts", size_dicts)

      #size_dicts = center_sizes
      #max_sizes = center_sizes[0]
      # max_sizes = center_sizes[0]

      # find largest structures in entire set to set alignment width
      # TODO: If not doing more advanced clustering, update this to max sizes per cluster
      #for key in max_sizes:
      #  for size_dict in size_dicts:
      #    max_sizes[key] = size_dict[key] if size_dict[key] > max_sizes[key] else max_sizes[key]

      #print(max_sizes)

      # multip = 1.5
      # for k in DYNAMIC_INTERACTION_KEYS:
      #   for s in center_sizes:
      #     # assign some buffer room
      #     s[k] = math.ceil(s[k] * multip)
      # max_sizes = center_sizes[0]
      for k in center_sizes[0].keys():
        for s in center_sizes:
          # max_sizes[k] = max(max_sizes[k], s[k])
          #temp until interaction counting is fixed
          if (k != 'num_atoms' and k != 'periodic_image_count'):
            s[k] = max(s[k], 300)
          #if k == 'filter3_size':
          #  s[k] = 1000
          #if k == 'filter4_size':
          #  s[k] = 23000
      #max_sizes = frozendict(max_sizes)
      #print("[INFO] Interaction list sizes:")
      #for item in max_sizes.items():
      #    print(item)

      # aligned_data = align_and_batch_structures(systems, max_sizes, batch_size=ff_and_geo_batch_size)
      # aligned_data = [move_dataclass(d, jnp) for d in aligned_data]

      # might have to do center_sizes[i] = max_sizes for 0 to len(clusters)
      # center_sizes = [max_sizes for i in aligned_data]
      #print("aligned_data len", len(aligned_data))
      #print("aligned_data -1 len", len(aligned_data[-1].name))
      #print("len center sizes", len(center_sizes))
      #print("Max sizes", max_sizes)
      #print("Center Sizes", center_sizes)
      #range()len()data???

      #print("[INFO] Building AMBER Parameter List")
      # TODO: this eventually needs to be rolled into a dataclass and some work will have to be done to create a treemap
      # for vmap compatibility, see: https://stackoverflow.com/questions/73765064/jax-vmap-over-batch-of-dataclasses
      # the aligning done to the list structures may also serve this purpose if there isn't an underlying mechanism that works
      #flist = build_prm_list(args.geo, args.init_FF_amber)
      #print("Flist", flist)
      #prm_dict_list, max_sizes = load_ff(flist)
      # for prm in prm_dict_list:
      #   for key, value in prm.items():
      #     print(key, value)
      # sys.exit()
      
      #aligned_amber_ff = align_forcefield(prm_dict_list, max_sizes)
      #ff_and_geo_batch_size = 40

      # aligned_amber_ff = align_and_batch_forcefield(prm_dict_list, max_sizes, ff_and_geo_batch_size)

      aligned_amber_ff = [move_dataclass(d, jnp) for d in aligned_amber_ff]
      #print("aligned ff len", len(aligned_amber_ff))
      #print("aligned ff len -1", len(aligned_amber_ff[-1].b_k))
      #print("type", type(aligned_amber_ff[0]))










  ## Alternate Clustering if using AMBER EEM
  # if(args.ff_type == 'ambereem'):
  #   print("[INFO] Starting AMBER Optimization w/ EEM")

  #   ff_and_geo_batch_size = 20

  #   num_threads = os.cpu_count()
  #   print("num_threads", num_threads)
  #   #update this to something like
  #   #https://stackoverflow.com/questions/76308447/how-to-use-properly-slurm-sbatch-and-python-multiprocessing
  #   #ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
  #   #should hopefully return the actual allocated number of cpus

  #   # [globally_sorted_indices,
  #   # all_cut_indices,
  #   # center_sizes] = process_and_cluster_geos(systems, force_field,
  #   #                                           max_num_clusters=args.max_num_clusters,
  #   #                                           num_threads=num_threads,
  #   #                                           chunksize=4,
  #   #                                           close_cutoff=5.0, far_cutoff=10.0)

  #   # print("Process and cluster done")

  #   #batched alloc scheme
  #   #find max periodic image counts and atom counts
  #   #cluster geos by 20
  #   #count interaction lists for each cluster in loop
  #   #ideally only count far nbr sizes or figure out alterantive far nbr scheme
  #   #would jax md neighbor lists work with indices passed to eem?
  #   #center sizes = each cluster's size
  #   def cluster_structs(structures, batch_size, dtype=jnp.float32):
  #     full_size = len(structures)
  #     max_sizes = {'num_atoms':0, 'periodic_image_count':0}
  #     for struct in structures:
  #       max_sizes['num_atoms'] = max(max_sizes['num_atoms'], len(struct.atom_types))
  #       max_sizes['periodic_image_count'] = len(struct.periodic_image_shifts)
  #     batches = []
  #     center_sizes = []
  #     for bs in range(0,abs(full_size-batch_size),batch_size):
  #       batch = align_structures(structures[bs:bs+batch_size],
  #                            max_sizes, dtype)
  #       batches.append(batch)
  #       center_sizes.append(count_inter_list_sizes(structures[bs:bs+batch_size], force_field, 
  #                                      num_threads=num_threads, chunksize=4,
  #                                      close_cutoff=5.0,
  #                                      far_cutoff=10.0))

  #     return batches, center_sizes

  #   aligned_data = cluster_structs(systems, ff_and_geo_batch_size)
  #   aligned_data = 

  #   #atom_types = structure.atom_types
  #   #num_atoms = len(atom_types)
  #   #size_dict["num_atoms"] = len(atom_mask)
  #   #size_dict["periodic_image_count"] = len(structure.periodic_image_shifts)




  #   size_dicts = count_inter_list_sizes(systems, force_field, 
  #                                      num_threads=num_threads, chunksize=4,
  #                                      close_cutoff=5.0,
  #                                      far_cutoff=10.0)
  #   print("Inter list size counting done")
  #   #print("center sizes", center_sizes)

  #   #sys.exit()


  #   #print("size_dicts", size_dicts)

  #   #size_dicts = center_sizes
  #   #max_sizes = center_sizes[0]
  #   max_sizes = size_dicts[0]

  #   # find largest structures in entire set to set alignment width
  #   # TODO: If not doing more advanced clustering, update this to max sizes per cluster
  #   #for key in max_sizes:
  #   #  for size_dict in size_dicts:
  #   #    max_sizes[key] = size_dict[key] if size_dict[key] > max_sizes[key] else max_sizes[key]

  #   #print(max_sizes)

  #   multip = 1.5
  #   for k in DYNAMIC_INTERACTION_KEYS:
  #     for s in size_dicts:
  #       # assign some buffer room
  #       s[k] = math.ceil(s[k] * multip)
  #   max_sizes = size_dicts[0]
  #   for k in max_sizes.keys():
  #     for s in size_dicts:
  #       max_sizes[k] = max(max_sizes[k], s[k])
  #       #temp until interaction counting is fixed
  #       if (k != 'num_atoms' and k != 'periodic_image_count'):
  #         max_sizes[k] = max(max_sizes[k], 200)
  #       if k == 'filter3_size':
  #         max_sizes[k] = 1000
  #       if k == 'filter4_size':
  #         max_sizes[k] = 23000
  #   #max_sizes = frozendict(max_sizes)
  #   print("[INFO] Interaction list sizes:")
  #   for item in max_sizes.items():
  #       print(item)

  #   aligned_data = align_and_batch_structures(systems, max_sizes, batch_size=ff_and_geo_batch_size)
  #   aligned_data = [move_dataclass(d, jnp) for d in aligned_data]

  #   # might have to do center_sizes[i] = max_sizes for 0 to len(clusters)
  #   center_sizes = [max_sizes for i in aligned_data]
  #   print("aligned_data len", len(aligned_data))
  #   #print("len center sizes", len(center_sizes))
  #   print("Max sizes", max_sizes)
  #   print("Center Sizes", center_sizes)
  #   #range()len()data???

  #   print("[INFO] Building AMBER Parameter List")
  #   # TODO: this eventually needs to be rolled into a dataclass and some work will have to be done to create a treemap
  #   # for vmap compatibility, see: https://stackoverflow.com/questions/73765064/jax-vmap-over-batch-of-dataclasses
  #   # the aligning done to the list structures may also serve this purpose if there isn't an underlying mechanism that works
  #   flist = build_prm_list(args.geo, args.init_FF_amber)
  #   #print("Flist", flist)
  #   prm_dict_list, max_sizes = load_ff(flist)
  #   # for prm in prm_dict_list:
  #   #   for key, value in prm.items():
  #   #     print(key, value)
  #   # sys.exit()
    
  #   #aligned_amber_ff = align_forcefield(prm_dict_list, max_sizes)
  #   #ff_and_geo_batch_size = 40

  #   aligned_amber_ff = align_and_batch_forcefield(prm_dict_list, max_sizes, ff_and_geo_batch_size)

  #   aligned_amber_ff = [move_dataclass(d, jnp) for d in aligned_amber_ff]
  #   print("aligned ff len", len(aligned_amber_ff))
  #   print("type", type(aligned_amber_ff[0]))

  # if(args.ff_type == 'amber'):
  #   num_threads = os.cpu_count()
  #   print("SLURM CPUS PER TASK", int(os.environ['SLURM_CPUS_PER_TASK']))
  #   [globally_sorted_indices,
  #   all_cut_indices,
  #   center_sizes] = process_and_cluster_geos(systems, force_field,
  #                                             max_num_clusters=args.max_num_clusters,
  #                                             num_threads=num_threads,
  #                                             chunksize=4,
  #                                             close_cutoff=5.0, far_cutoff=10.0)
  #   for i in range(len(center_sizes)):
  #       for k in center_sizes[i].keys():
  #           if k in DYNAMIC_INTERACTION_KEYS:
  #             multip = 1.5
  #             # give extra buffer room if we need to e. minim
  #             if (k in ['filter3_size', 'filter4_size']
  #                 and args.num_e_minim_steps > 0):
  #               multip = 2.0
  #             center_sizes[i][k] = math.ceil(multip * center_sizes[i][k])
  #             if (k != 'num_atoms' and k != 'periodic_image_count'):
  #               center_sizes[i][k] = max(center_sizes[i][k], 200)
  #             if k == 'far_nbr_size':
  #               center_sizes[i][k] = 300
  #             if k == 'filter3_size':
  #               center_sizes[i][k] = 1000
  #             if k == 'filter4_size':
  #               center_sizes[i][k] = 23000
  #           if center_sizes[i][k] == 0:
  #               center_sizes[i][k] = 1
  #   for i in range(len(center_sizes)):
  #     for item in center_sizes[i].items():
  #       print(item)

  #   aligned_data = []
  #   for i in range(len(center_sizes)):
  #       zz = align_structures([systems[i] for i in all_cut_indices[i]], center_sizes[i], TYPE)
  #       zz = move_dataclass(zz, jnp)
  #       aligned_data.append(zz)

  #   print("[INFO] Building AMBER Parameter List")
  #   flist = build_prm_list(args.geo, args.init_FF_amber)
  #   #prm_dict_list, max_sizes = load_ff(flist)
  #   aligned_amber_ff = []
  #   for i in range(len(center_sizes)):
  #     zz, max_size = load_ff([flist[i] for i in all_cut_indices[i]])
  #     zz = align_forcefield(zz, max_size)
  #     zz = move_dataclass(zz, jnp)
  #     aligned_amber_ff.append(zz)


  # else:
  # ###########################################################################
  #   num_threads = os.cpu_count()
  #   [globally_sorted_indices,
  #   all_cut_indices,
  #   center_sizes] = process_and_cluster_geos(systems, force_field,
  #                                             max_num_clusters=args.max_num_clusters,
  #                                             num_threads=num_threads,
  #                                             chunksize=4,
  #                                             close_cutoff=5.0, far_cutoff=10.0)
  #   for i in range(len(center_sizes)):
  #       for k in center_sizes[i].keys():
  #           if k in DYNAMIC_INTERACTION_KEYS:
  #             multip = 1.5
  #             # give extra buffer room if we need to e. minim
  #             if (k in ['filter3_size', 'filter4_size']
  #                 and args.num_e_minim_steps > 0):
  #               multip = 2.0
  #             center_sizes[i][k] = math.ceil(multip * center_sizes[i][k])
  #           if center_sizes[i][k] == 0:
  #               center_sizes[i][k] = 1

  #   aligned_data = []
  #   for i in range(len(center_sizes)):
  #       zz = align_structures([systems[i] for i in all_cut_indices[i]], center_sizes[i], TYPE)
  #       zz = move_dataclass(zz, jnp)
  #       aligned_data.append(zz)

  force_field = move_dataclass(force_field, jnp)

  

  batched_allocate, batched_allocate_amber = reaxff_interaction_list_generator(force_field,
                                                       close_cutoff = 5.0,
                                                       far_cutoff = 10.0,
                                                       use_hbond=True)
  
  #alternative far nbr generation
  # displacement_fn, shift_fn = jax_md.space.periodic(50.0)
  # neighbor_fn2 = jax_md.partition.neighbor_list(displacement_fn,
  #                                 box=50.0,
  #                                 r_cutoff=10.0,
  #                                 dr_threshold=0.5,
  #                                 capacity_multiplier=1.2,
  #                                 format=jax_md.partition.Dense)

  if(args.ff_type == 'ambereem'):
    allocate_func = jax.jit(batched_allocate_amber,static_argnums=(3,))
    # allocate_func = neighbor_fn2.allocate
  else:
    allocate_func = jax.jit(batched_allocate,static_argnums=(3,))
  center_sizes = [frozendict(c) for c in center_sizes]

  list_positions = [s.positions for s in aligned_data]

  get_params_jit = jax.jit(get_params,static_argnums=(1,))
  set_params_jit = jax.jit(set_params,static_argnums=(1,))

  # TODO: Fix the JIT behavior of these functions by fixing the underlying vmapping
  force_f = jax.jit(jax.vmap(jax.value_and_grad(calculate_energy_and_charges_w_rest,
                                            has_aux=True),
                         in_axes=(0,0,0,None,0,None)), static_argnames=('ff_type_int'))
  
  # force_f = jax.vmap(jax.value_and_grad(calculate_energy_and_charges_w_rest,
  #                                            has_aux=True),
  #                         in_axes=(0,0,0,None,0,None))

  minimize_kwargs = {"allocate_func":allocate_func, "force_func":force_f,
                     "init_LR":args.e_minim_LR, "minim_steps":args.num_e_minim_steps
                     , "target_RMSG":args.end_RMSG}
  minim_func = partial(energy_minimize, **minimize_kwargs)


  loss_and_grad_func = jax.jit(jax.value_and_grad(calculate_loss),
                               static_argnames=('return_indiv_error','ff_type_int'))
  loss_func = jax.jit(calculate_loss, static_argnames=('return_indiv_error','ff_type_int'))
  # loss_and_grad_func = jax.value_and_grad(calculate_loss)
  # loss_func = calculate_loss

  # for i in range(len(aligned_data)):
  #   sub_nbr, counts, overflow = batched_allocate_amber(list_positions[i], aligned_data[i],
  #                                  force_field, center_sizes[i])
  #   inds = sub_nbr[0]
  #   print("sub nbr len", inds.shape)
  #   print("inds 1", inds[0])
  #   print("counts", counts)

  # inds 1 [[ 1  2  3 ... 23 23 23]
  # [ 0  2  3 ... 23 23 23]
  # [ 0  1  3 ... 23 23 23]
  # ...
  # [ 0  1  2 ... 23 23 23]
  # [ 0  1  2 ... 23 23 23]
  # [ 0  1  2 ... 23 23 23]]


  # nbs = jnp.array([neighbor_fn2.allocate(pos).idx for pos in list_positions[0]])
  # print("nbs shape", nbs.shape)

  # sys.exit()
  

  # #aligned_data
  # #aligned_amber_forcefield
  # #flist
  # list_positions = [s.positions for s in aligned_data]
  # list_positions = list_positions[0] # 18,27,3 shape
  # list_positions = onp.array(list_positions)
  # #print(len(aligned_amber_ff))
  # for i in range(18):
  #   print(flist[i])
  #   inpcrd = omm.app.AmberInpcrdFile(flist[i][:-7]+'/inpcrd')
  #   prmtop = omm.app.AmberPrmtopFile(flist[i])
  #   positions = list_positions[i]/10 # A -> NM
  #   amberPrms = aligned_amber_ff
  #   #print(prmtop._prmtop._raw_data['POINTERS'][0])
  #   natom = int(prmtop._prmtop._raw_data['POINTERS'][0])
  #   bondprm = (amberPrms.b_k[i], amberPrms.b_l[i], amberPrms.b_1_idx[i], amberPrms.b_2_idx[i], amberPrms.b_prm_idx[i])
  #   angleprm = (amberPrms.a_k[i], amberPrms.a_eq_ang[i], amberPrms.a_1_idx[i], amberPrms.a_2_idx[i], amberPrms.a_3_idx[i], amberPrms.a_prm_idx[i])
  #   torsionprm = (amberPrms.t_k[i], amberPrms.t_phase[i], amberPrms.t_period[i], amberPrms.t_1_idx[i], amberPrms.t_2_idx[i], amberPrms.t_3_idx[i], amberPrms.t_4_idx[i], amberPrms.t_prm_idx[i])
  #   ljprm = (amberPrms.pairs[i], amberPrms.pairs14[i], amberPrms.lj_type[i], amberPrms.sigma[i], amberPrms.epsilon[i], amberPrms.scnb[i])
  #   coulprm = (amberPrms.charges[i], amberPrms.pairs[i], amberPrms.pairs14[i], amberPrms.scee[i])

  #   #get point energies and forces from jax
  #   boxVectors = jnp.array([100.0,100.0,100.0])
  #   def en_fn(pos):
  #     totalE = 0
  #     totalE += amber.bond_get_energy(pos, boxVectors, bondprm)
  #     totalE += amber.angle_get_energy(pos, boxVectors, angleprm)
  #     totalE += amber.torsion_get_energy(pos, boxVectors, torsionprm)
  #     totalE += amber.lj_get_energy(pos, boxVectors, ljprm)
  #     totalE += amber.coul_get_energy(pos, boxVectors, coulprm)
  #     return totalE

  #   jax_nrg = en_fn(positions)
  #   print("JAX Energy", jax_nrg)

  #   #get point energies and forces from omm
  #   system = prmtop.createSystem(nonbondedMethod=omm.app.NoCutoff, removeCMMotion=False)
  #   #boxVectors = jnp.array([v._value for v in system.getDefaultPeriodicBoxVectors()])
  #   #boxVectors = boxVectors.sum(axis=0)
  #   integrator = omm.VerletIntegrator(0.001*omm.unit.picoseconds)
  #   simulation = omm.app.Simulation(prmtop.topology, system, integrator)
  #   #print(positions.shape)
  #   #print(positions[:natom])
  #   simulation.context.setPositions(positions[:natom])
  #   #simulation.context.setPositions(inpcrd.positions)
  #   state = simulation.context.getState(getEnergy=True, getForces=True)

  #   omm_forces = state.getForces(asNumpy=True)
  #   omm_nrg = state.getPotentialEnergy()._value

  #   print("OpenMM Energy", omm_nrg)
  #   print("Energy Differences (Absolute , Percentage):", abs(jax_nrg-omm_nrg), ',', abs(abs(jax_nrg-omm_nrg)/jax_nrg)*100, '%')

  #   #compare and find percent deviation, rms, etc
  #   g_en_fn = jax.grad(en_fn)
  #   jax_forces = g_en_fn(positions)[:natom]

  #   print("Force Differnces (Percentage-Maximum Component)")
  #   diff = []
  #   #print("frc len", omm_forces.shape)
  #   #print("jax len", jax_forces.shape)
  #   for i in range(len(jax_forces)):
  #     jax_frc = jnp.linalg.norm(jax_forces[i])
  #     omm_frc = jnp.linalg.norm(omm_forces[i]._value)
  #     diff.append(abs(jax_frc-omm_frc)/jax_frc)
  #   print(max(diff))

  # sys.exit()

  def new_loss_and_grad_func(params, param_indices,
                             force_field, training_data,
                             list_positions, aligned_data, center_sizes,
                             amberPrms=None, ff_type_int=None):
    params = jnp.array(params)
    force_field = set_params_jit(force_field, param_indices, params)
    all_inters = [allocate_func(list_positions[i], aligned_data[i],
                                force_field, center_sizes[i])[0]
                  for i in range(len(center_sizes))]
    loss, grads_ff = loss_and_grad_func(force_field,
                                        list_positions,
                                        aligned_data,
                                        all_inters,
                                        training_data,
                                        False,
                                        amberPrms,
                                        ff_type_int)

    grads = get_params_jit(grads_ff, param_indices)
    loss = onp.asarray(loss,dtype=onp.float64)
    grads = onp.asarray(grads,dtype=onp.float64)

    return loss, grads

  def new_loss_func(params, param_indices,
                    force_field, training_data,
                    list_positions, aligned_data, center_sizes,
                    return_indiv_error = False,
                    amberPrms=None, ff_type_int=None):
    params = jnp.array(params)
    force_field = set_params_jit(force_field, param_indices, params)
    all_inters = [allocate_func(list_positions[i], aligned_data[i],
                                force_field, center_sizes[i])[0]
                  for i in range(len(center_sizes))]
    results = loss_func(force_field,
                    list_positions,
                    aligned_data,
                    all_inters,
                    training_data,
                    return_indiv_error,
                    amberPrms,
                    ff_type_int)
    if return_indiv_error:
      loss, indiv_errors = results
    else:
      loss = results
    loss = onp.asarray(loss, dtype=onp.float64)
    if return_indiv_error:
      return loss, indiv_errors
    return loss

  init_params = get_params(force_field, param_indices)
  init_params = onp.array(init_params)


  population_size = args.num_trials
  random_sample_count = args.random_sample_count
  results_list = []
  best_params = None
  best_fitness = float("inf")
  opt_method = args.opt_method
  num_steps = args.num_steps
  e_minim_flag = sum([jnp.sum(data.energy_minimize) for data in aligned_data]) > 0
  e_minim_flag = e_minim_flag & (args.num_e_minim_steps > 0)
  if opt_method == "L-BFGS-B":
      optim_options =dict(maxiter=100,maxls=20,maxcor=20, disp=False)
  else:
      optim_options =dict(maxiter=100, disp=False)

  for i in range(population_size):
    print('*' * 40)
    print("Trial-{} is starting...".format(i+1))
    start = time.time()
    if args.init_FF_type == 'random':
      min_params = random_parameter_search(bounds, random_sample_count,
                                  param_indices, force_field, training_data,
                                  list_positions, aligned_data, center_sizes,
                                  new_loss_func)
      selected_params = min_params
    elif args.init_FF_type == 'educated':
      selected_params = add_noise_to_params(init_params, bounds, scale=0.1)
    else: # fixed
      selected_params = jnp.array(init_params)

    [global_min_params,
     global_min,
     center_sizes] = train_FF(selected_params, param_indices, bounds, force_field,
                           aligned_data, center_sizes, training_data,
                           validation_data,
                           num_steps, e_minim_flag, opt_method, optim_options,
                           advanced_opts,
                           new_loss_and_grad_func, minim_func, allocate_func,
                           # None, #args.ff_type,
                           aligned_amber_ff,
                           ff_type_int)
    end = time.time()

    result = {"time":end-start, "value": global_min,
              "params": global_min_params,
              "unique_id":i+1}
    results_list.append(result)

    if best_fitness > global_min or best_params == None:
      best_fitness = global_min
      best_params = global_min_params

    print("Trial-{} ended, loss value: {:.2f}".format(i+1, global_min))
    print("Lowest loss so far        : {:.2f}".format(best_fitness))


  if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

  if args.save_opt == "all":
    results_to_save = results_list
  else:
    results_to_save = [{'params':best_params, 'value':best_fitness,
                        "unique_id":"best"}]
  if population_size <= 0:
     print("[INFO] The population size <= 0, the initial force field is being evaluated...")
     results_to_save = [{'params':jnp.array(init_params), 'value':float('inf'),
                        "unique_id":"init_ff"}]

  for ii,res in enumerate(results_to_save):
    params = res['params']
    current_loss = res['value']
    unique_id = res['unique_id']
    force_field = set_params_jit(force_field, param_indices, params)
    if e_minim_flag:
      minim_start = time.time()
      [list_positions, cur_total_energy,
      center_sizes, cur_RMSG_vals] = minim_func(aligned_data,
                                                center_sizes,
                                                force_field,
                                                amberPrms=aligned_amber_ff,
                                                ff_type_int=ff_type_int)
      minim_end = time.time()
    else:
      # extend the interaction list sizes if needed
      for i in range(len(aligned_data)):
        sub_nbr = allocate_func(list_positions[i], aligned_data[i],
                                   force_field, center_sizes[i])[0]
        if jnp.any(sub_nbr.did_buffer_overflow):
          center_sizes[i] = update_inter_sizes(list_positions[i],
                                                   aligned_data[i],
                                                   force_field,
                                                   center_sizes[i],
                                                   multip=1.5)

    loss, indiv_errors = new_loss_func(params, param_indices,
                                      force_field, training_data,
                                      list_positions, aligned_data,
                                      center_sizes,
                                      True, aligned_amber_ff, ff_type_int)
    for k in indiv_errors.keys():
      # move data to regular numpy arrays
      for i,sub_val in enumerate(indiv_errors[k]):
        indiv_errors[k][i] = onp.array(sub_val)
    loss = float(loss)
    loss_str = str(round(loss))
    new_name = "{}/new_FF_{}_{}".format(args.out_folder,unique_id,loss_str)
    new_force_field = move_dataclass(force_field, onp)
    parse_and_save_force_field(args.init_FF, new_name, new_force_field)

    report_name = "{}/report_{}_{}.txt".format(args.out_folder,unique_id,loss_str)
    produce_error_report(report_name, training_data, indiv_errors, geo_index_to_name)

    # produce the report for the validation data if available
    if args.use_valid:
      [valid_loss,
       valid_indiv_errors] = new_loss_func(params, param_indices,
                                        force_field, validation_data,
                                        list_positions, aligned_data,
                                        center_sizes,
                                        True, aligned_amber_ff, ff_type_int)
      for k in valid_indiv_errors.keys():
        # move data to regular numpy arrays
        for i,sub_val in enumerate(valid_indiv_errors[k]):
          valid_indiv_errors[k][i] = onp.array(sub_val)
      valid_loss = float(valid_loss)
      valid_loss_str = str(round(valid_loss))
      report_name = "{}/valid_report_{}_{}.txt".format(args.out_folder,unique_id,valid_loss_str)
      produce_error_report(report_name, validation_data, valid_indiv_errors, geo_index_to_name)

if __name__ == "__main__":
  main()
