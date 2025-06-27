#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver code to run the trainer

Author: Mehmet Cagri Kaymak, William Betancourt
"""
import os
import sys
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
import jax.profiler
import jax.numpy as jnp
import numpy as onp
import time
import argparse
from .smartformatter import SmartFormatter
from frozendict import frozendict
from jax_md.reaxff.reaxff_energy import calculate_reaxff_energy
from jax_md.reaxff.reaxff_forcefield import ForceField
from jax_md.reaxff.reaxff_helper import read_force_field
from jax_md import dataclasses
from jaxreaxff.optimizer import (calculate_loss, 
                                 calculate_energy_and_charges_w_rest, 
                                 add_noise_to_params, random_parameter_search, 
                                 train_FF, energy_minimize, update_inter_sizes) 
from jaxreaxff.helper import set_params, get_params, produce_error_report
from jaxreaxff.interactions import (reaxff_interaction_list_generator, 
                                    calculate_dist_and_angles, 
                                    DYNAMIC_INTERACTION_KEYS)
from jaxreaxff.structure import align_structures
from jaxreaxff.helper import (move_dataclass, process_and_cluster_geos, 
                              create_structure_map, read_parameter_file, 
                              map_params, read_geo_file, read_train_set, 
                              filter_data, structure_training_data,
                              parse_and_save_force_field)
import math
from functools import partial
from jaxreaxff.helper import build_float_range_checker
from jaxreaxff.structure_amber import (load_amber_ff_batch, map_params_amber,
                                       process_and_cluster_geos_amber,
                                       process_and_cluster_ff_amber,
                                       align_ff_amber, parse_and_save_force_field_amber)
from jaxreaxff.helper_prmtop import build_prm_list
from jax_md.amber.amber_energy_v2 import amber_energy
from jax_md.amber.amber_helper import load_amber_ff
from scipy import optimize
from scipy.stats import qmc
import evosax

# TODO look through argparse documentation and add any relevant information, also potentially separate into module
def main():
  # create parser for command-line arguments
  parser = argparse.ArgumentParser(description='JAX-ParamOpt driver',
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
      choices=['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr'],
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

  def validate_init_type(value):
    closed_choices = ['random', 'educated', 'fixed', 'cmaes', 'openes', 'pgpe', 'snes', 'lga', 'diffev', 'shgo', 'direct', 'basin', 'sobol', 'adaptive']
    if value in closed_choices:
      return value # valid choice from closed list

    # check to see if the method is a valid GA
    elif value.startswith("genetic_"):
      function_name = value[len("genetic_"):]
      try:
        # try to get the function dynamically
        func = getattr(evosax.algorithms, function_name)
        return value  # valid algo function name
      except AttributeError:
        raise argparse.ArgumentTypeError(f"[ERROR] Function '{function_name}' not found in 'evosax.algorithms' module.")
    else:
        raise argparse.ArgumentTypeError(f"[ERROR] Invalid input: '{value}'. Must be one of the closed options or start with 'genetic_'.")

  parser.add_argument('--init_FF_type', metavar='init_type',
      type=validate_init_type,
      # choices=['random', 'educated', 'fixed', 'cmaes', 'openes', 'pgpe', 'snes', 'lga', 'diffev', 'shgo', 'direct', 'basin'], #TODO add missing tips here
      # TODO read up on the wikipedia page for global optimization https://en.wikipedia.org/wiki/Global_optimization#Deterministic_methods
      # need to think about evolutionary algorithms, swarm intelligence, bayesian optimization, deterministic global search (e.g. direct),
      # and something like simulated annealing or other multistart local search methods, maybe with low discrepancy sequence selection
      # for GA could do some template to the effect of ga_cmaes and then filter based on this
      default='fixed',
      help='R|How to start the trials from the given initial force field.\n' +
      '"random": Sample the parameters from uniform distribution between given ranges.\n' +
      '"educated": Sample the parameters from a narrow uniform distribution centered at given values.\n' +
      '"fixed": Start from the parameters given in "init_FF" file') # TODO fill in the new options
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
  parser.add_argument('--bonded_cutoff', metavar='cutoff',
      type=float,
      default=5.0,
      help='Cutoff distance for bonded interactions (in Angstrom).')
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
  parser.add_argument('--perc_noise_when_stuck', metavar='percentage',
      type=build_float_range_checker(0.0, 0.1),
      default=0.04,
      help='R|Percentage of the noise that will be added to the parameters\n' +
           'when the optimizer is stuck.\n' +
           'param_noise_i = (param_min_i, param_max_i) * perc_noise_when_stuck\n' +
           'Allowed range: [0.0, 0.1]')
  parser.add_argument('--seed', metavar='seed',
      type=int,
      default=0,
      help='Seed value')
  # AMBER related parameters
  parser.add_argument('--ff_type', metavar='ff_type',
      choices=['reaxff', 'amber', 'ambereem'],
      type=str,
      default='reaxff',
      help='Forcefield to optimize - "reaxff" or "amber" or "ambereem"')
  parser.add_argument('--ffq_params', metavar='filename',
      type=str,
      default="ffq_params",
      help='Supplemental parameter file for FFQ if AMBER is enabled')
  parser.add_argument('--opt_mode', metavar='mode',
      choices=['single', 'group'],
      type=str,
      default="group",
      help='For AMBER, determines optimization type\n' +
           'single - optimization is for a single system\n' +
           'group - optimization is done for a group of files that\n' +
           'share common parameters defined in the params file')
  parser.add_argument('--min_type', metavar='min_type',
      choices=['grad', 'fire', 'dlfind', 'bfgs'], # TODO add fire description
      type=str,
      default='grad',
      help='Method to use for energy minimization - "grad" or "dlfind" or "bfgs"\n' +
           '"dlfind" uses internal coordinates and calls libdlfind library' +
           '"grad" uses gradient descent internally to optimize structures' +
           '"bfgs" uses Scipy L-BFGS-B to optimize strucutres')
  # TODO consider if this is necessary, the idea is to identify the minimum (instead of a fixed 0 point) to normalize energy values
  parser.add_argument('--relative_energies', metavar='boolean',
    type=bool,
    default=False,
    help='Flag indicating whether to use relative energies or absolute energies for loss evaluation.\n'+
         'Relative energies are useful for things like torsion scanning.')
  parser.add_argument('--debug_level', metavar='level',
      type=int,
      default=0,
      choices=range(0, 10),
      help='The debug/reporting level that the optimizer will run\n' +
           'Many of the higher levels use JAX callbacks that are slow\n' +
           '0 - (default) No debugging information\n' +
           '1 - More timing information printed per loop\n' +
           '2 - Per component loss information\n' +
           '3 - Per component, per structure loss information\n' +
           '4 - Full loss gradient information\n' +
           '5 - Full dump of internal optimizer state to hdf5 log') # TODO need to do energy components, neighbor lists, forces, loss gradients
  
  # TODO clean this up after looking at amber/gaussian/genesis for log template
  # motd =  "***********************************************************************\n" +
  #         "|                                                                     |\n" +
  #         "|                     ----------------------------                    |\n" +
  #         "|                     |       JAX-ParamOpt       |                    |\n" +
  #         "|                     |                          |                    |\n" +
  #         "|                     |    Auto-differentiable   |                    |\n" +
  #         "|                     |   Chemical Force Field   |                    |\n" +
  #         "|                     |  Parameter Optimization  |                    |\n" +
  #         "|                     ----------------------------                    |\n" +
  #         "|                                                                     |\n" +
  #         "|            Developed and maintained by William Betancourt           |\n" +
  #         "|                                                                     |\n" +
  #         "|        Fork of the JAX-ReaxFF codebase by Mehmet Cagri Kaymak       |\n" +
  #         "|                                                                     |\n" +
  #         "|                 *PUBLICATION REFERENCE JAX-PARAMOPT*                |\n" +
  #         "|                  *PUBLICATION REFERENCE JAX-REAXFF*                 |\n" +
  #         "|                    *JAX MD/JAX/OTHER REFERENCES*                    |\n" +
  #         "|                                                                     |\n" +
  #         "|                      Michigan State University                      |\n" +
  #         "|                      Department of Engineering                      |\n" +
  #         "|                                                                     |\n" +
  #         "***********************************************************************\n"
  # print(motd)

  # TODO add debug argument here that dumps pre and post geo opt energies/components
  # maybe also print or dump more internal data structures
  # orgainze into debug levels or sections
  # e.g. 1 dumps component losses, 2 dumps component energies for min,

  #parse arguments
  args = parser.parse_args()
  # TODO: remove
  args.save_opt = "all"
  default_backend = jax.default_backend().lower()
  
  if default_backend == 'cpu':
    print("[WARNING] Falling back to CPU")
    print("To use the GPU version, jaxlib with CUDA support needs to installed!")
  
  # advanced options
  advanced_opts = {"perc_err_change_thr":0.01,                         # if change in error is less than this threshold, add noise
                   "perc_noise_when_stuck":args.perc_noise_when_stuck, # noise percantage (wrt param range) to add when stuck
                   "perc_width_rest_search":0.15,                      # width of the restricted parameter search after iteration > rest_search_start
                   }
  
  if args.ff_type == "reaxff" and not args.min_type == "grad":
    print("[ERROR] Only currently supported geometry optimization for ReaxFF is grad")
    sys.exit()
  elif args.ff_type == "ambereem" and not args.min_type == "dlfind":
    print("[ERROR] Only currently supported geometry optimization for AMBEREEM is dlfind")
    sys.exit()
  elif args.ff_type == "amber" and not args.min_type == "dlfind":
    print("[ERROR] Only currently supported geometry optimization for AMBER is dlfind")
    sys.exit()

  onp.random.seed(args.seed)
  TYPE = jnp.float64
  # read the initial force field
  if args.ff_type == "reaxff":
    force_field = read_force_field(args.init_FF, cutoff2 = args.cutoff2, dtype=TYPE)
    force_field = ForceField.fill_off_diag(force_field)
    force_field = ForceField.fill_symm(force_field)
    ffq_ff = None
  # elif args.ff_type == "amber":
  #   print("[ERROR] Normal AMBER not implemented yet")
  #   sys.exit()
  elif args.ff_type == "ambereem":
    f_list = build_prm_list(args.geo, args.init_FF)
    force_fields, ffq_ff = load_amber_ff_batch(f_list, args.ffq_params, args.ff_type, dtype=TYPE)
  elif args.ff_type == "amber":
    #TODO more logic will need to be added for bespoke mode
    # the vectorization for this is already done in the ambereem option but the
    # parameter updates in that case are still only for eem parameters
    # how to handle equality constraints in the case of bespoke parameters?

    # really you have a case that is more like single or many and bespoke or dat/global
    # affdo torsions use a single ff but don't really try to update the underlying template
    # normal opt case would directly modify the .dat templates
    # so really you could have either single or many while still doing global
    # e.g. single prmtop but updating the gaff templates versus just updating the prmtop entries
    # the first case probably doesn't matter as much as you're just doing individual jobs at that point
    # really for bespoke in general

    print("[INFO] Loading AMBER parameters in single format")
    print("[WARNING] bespoke format with individual prmtops is currently not supported")
    print("[INFO] Nonbonded method is NoCutoff")

    # TODO no toggle for pme, charge method, dr threshold currently present
    force_field = load_amber_ff(inpcrd_file=None, prmtop_file=args.init_FF, 
                        ffq_file=None, nonbonded_method="NoCutoff",
                        charge_method="GAFF", dr_threshold=0.0, dtype=TYPE)

    ffq_ff = None
  
  # print INFO
  print("[INFO] Force field is read")
  ###########################################################################
  #read the paramemters to be optimized
  if args.ff_type == "reaxff":
    params_list_orig = read_parameter_file(args.params, ignore_sensitivity=0)
    params_list = map_params(params_list_orig, force_field.params_to_indices)
  elif args.ff_type == "ambereem" or args.ff_type == "amber":
    params_list_orig = read_parameter_file(args.params, ignore_sensitivity=0)
    params_list = map_params_amber(params_list_orig)

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
  if args.ff_type == "reaxff":
    systems = read_geo_file(args.geo, force_field.name_to_index, args.ff_type)
  elif args.ff_type == "ambereem" or args.ff_type == "amber":
    systems = read_geo_file(args.geo, None, args.ff_type)
  
  print("[INFO] Geometries have been read; count =", len(systems))

  if os.path.splitext(args.geo)[-1] in ['.h5', '.hdf5']:
    #TODO clean this up
    training_data = read_train_set(args.geo)
  else:
    training_data = read_train_set(args.train_file)
  # default value for the valid. data
  validation_data = None
  systems_tr, training_data = filter_data(systems, training_data)
  # read and process the validation data if used
  if args.use_valid:
    #print("[INFO] Validation data is provided!")
    #TODO fix this when testing data becomes available
    print("[ERROR] Validation data is not yet supported")
    sys.exit()
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

  print("[INFO] Training and validation data have been read")    
  geo_name_to_index, geo_index_to_name = create_structure_map(systems)
  training_data = structure_training_data(training_data, geo_name_to_index)
  if args.use_valid:
     validation_data = structure_training_data(validation_data, geo_name_to_index)
  # replace names with indices
  for i,s in enumerate(systems):
      s = dataclasses.replace(s, name = geo_name_to_index[s.name])
      systems[i] = s
  
  print("[INFO] Training data has been structured")  

  start_time = time.time()
  if args.ff_type == "reaxff":
    num_threads = os.cpu_count()    
    [globally_sorted_indices, 
    all_cut_indices, 
    center_sizes] = process_and_cluster_geos(systems, force_field,
                                              max_num_clusters=args.max_num_clusters, 
                                              num_threads=num_threads, 
                                              chunksize=4,
                                              close_cutoff=args.bonded_cutoff, far_cutoff=10.0)
    for i in range(len(center_sizes)):
        for k in center_sizes[i].keys():
            if k in DYNAMIC_INTERACTION_KEYS:
              multip = 1.5
              # give extra buffer room if we need to e. minim
              if (k in ['filter3_size', 'filter4_size'] 
                  and args.num_e_minim_steps > 0):
                multip = 2.0
              center_sizes[i][k] = math.ceil(multip * center_sizes[i][k])
            if center_sizes[i][k] == 0:
                center_sizes[i][k] = 1
  elif args.ff_type == "ambereem" or args.ff_type == "amber":
    batch_size = int(os.environ.get("SLURM_NTASKS", "1"))
    print("[INFO] Batch size for structure clustering is", batch_size)
    # TODO print batch size and other information about detected setup at top of file
    # also implement better clustering, this is just a placeholder
    # also make sure max num clusters is respected
    [all_cut_indices, 
    center_sizes] = process_and_cluster_geos_amber(systems, batch_size=batch_size, dtype=TYPE)

  end_time = time.time()
  print("[INFO] Geometries have been clustered; time =", end_time-start_time)

  aligned_data = []
  for i in range(len(center_sizes)):
      zz = align_structures([systems[i] for i in all_cut_indices[i]], center_sizes[i], TYPE)
      zz = move_dataclass(zz, jnp)
      aligned_data.append(zz)
  
  if args.ff_type == "reaxff":
    force_field = move_dataclass(force_field, jnp)
  elif args.ff_type == "ambereem":
    all_cut_indices_ff, center_sizes_ff = process_and_cluster_ff_amber(force_fields, batch_size=batch_size, dtype=TYPE)
    # TODO should move ff generation to onp
    aligned_ff = []
    for i in range(len(center_sizes_ff)):
      zz = align_ff_amber([force_fields[i] for i in all_cut_indices_ff[i]], center_sizes_ff[i], TYPE)
      zz = move_dataclass(zz, jnp)
      aligned_ff.append(zz)

    force_field = aligned_ff # TODO consider if there should just be different code paths for this
  elif args.ff_type == "amber":
    force_field = move_dataclass(force_field, jnp)

  if args.ff_type == "reaxff":
    batched_allocate = reaxff_interaction_list_generator(force_field,
                                                        close_cutoff = args.bonded_cutoff,
                                                        far_cutoff = 10.0,
                                                        use_hbond=True)
    
    allocate_func = jax.jit(batched_allocate,static_argnums=(3,))
  elif args.ff_type == "ambereem" or args.ff_type == "amber":
    # TODO this may be necessary to implement depending on how i want to
    # detach neighbor and other generation in the future
    allocate_func = lambda *args: None # or use an empty lambda

  center_sizes = [frozendict(c) for c in center_sizes]   
  
  list_positions = [s.positions for s in aligned_data]

  get_params_jit = jax.jit(get_params,static_argnums=(1,2,3))
  set_params_jit = jax.jit(set_params,static_argnums=(1,3,4))
  
  force_f = jax.jit(jax.vmap(jax.value_and_grad(calculate_energy_and_charges_w_rest,
                                            has_aux=True),
                         in_axes=(0,0,0, 0, None, None)), static_argnames=("ff_type"))
  
  minimize_kwargs = {"allocate_func":allocate_func, "force_func":force_f,
                     "init_LR":args.e_minim_LR, "minim_steps":args.num_e_minim_steps
                     , "target_RMSG":args.end_RMSG, "ff_type":args.ff_type}
  minim_func = partial(energy_minimize, **minimize_kwargs)
  
  if args.ff_type == "reaxff":
    loss_and_grad_func = jax.jit(jax.value_and_grad(calculate_loss, allow_int=True), # TODO consider if this is safe or how cagri avoided using this
                                static_argnames=('return_indiv_error','ff_type')) # TODO should this include ff_type?
  elif args.ff_type == "amber":
    loss_and_grad_func = jax.jit(jax.value_and_grad(calculate_loss, allow_int=True), # TODO consider if this is safe or how cagri avoided using this
                                static_argnames=('return_indiv_error','ff_type')) # TODO should this include ff_type?
  elif args.ff_type == "ambereem":
    loss_and_grad_func = jax.jit(jax.value_and_grad(calculate_loss, argnums=7, allow_int=True),
                                static_argnames=('return_indiv_error','ff_type'))
  loss_func = jax.jit(calculate_loss, static_argnames=('return_indiv_error','ff_type'))
  
  
  def new_loss_and_grad_func(params, param_indices,
                             force_field, training_data,
                             list_positions, aligned_data, center_sizes, ff_type, opt_mode, ffq_ff):
    params = jnp.array(params)
    if ff_type == "reaxff":
      force_field = set_params_jit(force_field, param_indices, params, ff_type, opt_mode)
    elif ff_type == "ambereem":
      ffq_ff = set_params_jit(ffq_ff, param_indices, params, ff_type, opt_mode)
    elif ff_type == "amber":
      force_field = set_params_jit(force_field, param_indices, params, ff_type, opt_mode)
    
    if ff_type == "reaxff":
      all_inters = [allocate_func(list_positions[i], aligned_data[i], 
                                force_field, center_sizes[i])[0] 
                  for i in range(len(center_sizes))]
    elif ff_type == "ambereem" or ff_type == "amber":
      all_inters = []

    loss, grads_ff = loss_and_grad_func(force_field,
                                        list_positions,
                                        aligned_data,
                                        all_inters,
                                        training_data,
                                        False,
                                        ff_type,
                                        ffq_ff)
  
    grads = get_params_jit(grads_ff, param_indices, ff_type, opt_mode)
    loss = onp.asarray(loss,dtype=onp.float64)
    grads = onp.asarray(grads,dtype=onp.float64)
  
    return loss, grads
  
  def new_loss_func(params, param_indices,
                    force_field, training_data,
                    list_positions, aligned_data, center_sizes, ff_type, opt_mode, ffq_ff,
                    return_indiv_error = False):
    params = jnp.array(params)
    if ff_type == "reaxff":
      force_field = set_params_jit(force_field, param_indices, params, ff_type, opt_mode)
    elif ff_type == "ambereem":
      ffq_ff = set_params_jit(ffq_ff, param_indices, params, ff_type, opt_mode)
    elif ff_type == "amber":
      force_field = set_params_jit(force_field, param_indices, params, ff_type, opt_mode)

    if ff_type == "reaxff":
      all_inters = [allocate_func(list_positions[i], aligned_data[i], 
                                  force_field, center_sizes[i])[0] 
                    for i in range(len(center_sizes))]
    elif ff_type == "ambereem" or ff_type == "amber":
      all_inters = []
    
    results = loss_func(force_field,
                    list_positions,
                    aligned_data,
                    all_inters,
                    training_data,
                    return_indiv_error,
                    ff_type,
                    ffq_ff)
    if return_indiv_error:
      loss, indiv_errors = results
    else:
      loss = results
    loss = onp.asarray(loss, dtype=onp.float64)
    if return_indiv_error:
      return loss, indiv_errors
    return loss
  if args.ff_type == "reaxff":
    init_params = get_params(force_field, param_indices, args.ff_type, args.opt_mode)
  elif args.ff_type == "ambereem":
    init_params = get_params(ffq_ff, param_indices, args.ff_type, args.opt_mode)
  elif args.ff_type == "amber":
    init_params = get_params(force_field, param_indices, args.ff_type, args.opt_mode)

  init_params = jnp.squeeze(init_params) # TODO figure out why this isn't necessary for cagri's code
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
                                  new_loss_func, args.ff_type, args.opt_mode, ffq_ff)
      selected_params = min_params
    elif args.init_FF_type == 'educated':
      selected_params = add_noise_to_params(init_params, bounds, scale=0.1)
    elif args.init_FF_type == 'fixed': # fixed
      selected_params = jnp.array(init_params)
    # Scipy based global optimization
    elif args.init_FF_type == 'sobol':
      args_loss = (param_indices, force_field, training_data,
          list_positions, aligned_data, center_sizes, args.ff_type, args.opt_mode, ffq_ff)
      sampler = qmc.Sobol(d=len(bounds), scramble=False)
      sample = sampler.random_base2(m=args.random_sample_count)
      # scale random distribution to bounds
      # TODO does this preserve the properties of the sequence?
      sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

      losses = jax.vmap(new_loss_func, in_axes=(0,None))(sample, *args_loss)
      best_loss_idx = jnp.argmin(losses)
      best_loss = losses[best_loss_idx]
      best_params = sample[best_loss_idx]
      sys.exit("sobol option not finished")
      # TODO what other relevant statistics to include for this?
      # mean, median, stdev, etc
      # lots of parameter guesses are completely unphysical so this is an interesting question

    elif args.init_FF_type == 'direct':
      print("doing direct opt")
      args_direct = (param_indices, force_field, training_data,
          list_positions, aligned_data, center_sizes, args.ff_type, args.opt_mode, ffq_ff)
      #TODO this likely requires more tuning due to memory issues
      #direct_options = dict(maxiter=10, maxls=20, maxcor=20, disp=False, jac=True) # maxiter 100
      direct_min_options = dict(method='L-BFGS-B')
      opt_bounds = optimize.Bounds(bounds[:,0], bounds[:,1])
      print("bounds", bounds)
      print("opt bounds", opt_bounds)
      opt_results = optimize.direct(new_loss_func, bounds=opt_bounds, args=args_direct,
                                  #options=direct_options,
                                  #minimizer_kwargs=direct_min_options
                                  )

      selected_params = opt_results.x

      jax.debug.print("opt results direct {}", opt_results)
    # elif args.init_FF_type == 'dual_annealing':
    # elif args.init_FF_type == 'differential_evolution':
    elif args.init_FF_type == 'shgo':
      args_shgo = (param_indices, force_field, training_data,
          list_positions, aligned_data, center_sizes, args.ff_type, args.opt_mode, ffq_ff)
      shgo_options = dict(maxiter=10, maxls=20, maxcor=20, disp=False, jac=True) # maxiter 100
      shgo_min_options = dict(method='L-BFGS-B')
      opt_results = optimize.shgo(new_loss_and_grad_func, bounds=bounds, args=args_shgo,
                                  options=shgo_options, minimizer_kwargs=shgo_min_options)

      jax.debug.print("opt results shgo {}", opt_results)

    # Code for GA routines ###########################################################################
    # TODO move to separate file eventually
    # elif args.init_FF_type == 'cmaes' or args.init_FF_type == 'pcmaes':      
    #   from evosax.algorithms import Sep_CMA_ES
    #   strategy = Sep_CMA_ES(population_size=256, solution)

    if args.init_FF_type.startswith("genetic_"):
      rng = jax.random.PRNGKey(0)
      args_loss = (param_indices, force_field, training_data,
          list_positions, aligned_data, center_sizes, args.ff_type, args.opt_mode, ffq_ff)
      initialization = jax.random.uniform(rng, (len(bounds),), minval=bounds[:,0], maxval=bounds[:,1])

      strategy_name = args.init_FF_type[len("genetic_"):]
      print("[INFO] Genetic algorithm is being used, strategy:", strategy_name)
      strategy_fn = getattr(evosax.algorithms, strategy_name)
      strategy = strategy_fn(population_size=256, solution=initialization)

    if(args.init_FF_type in ['cmaes','snes','openes','pgpe']):
      #rng = jax.random.PRNGKey(int(time.time()))
      rng = jax.random.PRNGKey(0)
      args_loss = (param_indices, force_field, training_data,
          list_positions, aligned_data, center_sizes, args.ff_type, args.opt_mode, ffq_ff)
      # es_params = strategy.default_params
      # state = strategy.initialize(rng, es_params)
      #TODO: consider replacing this with a random distribution
      initialization = jax.random.uniform(rng, (len(bounds),), minval=bounds[:,0], maxval=bounds[:,1])

      if args.init_FF_type == 'cmaes':
        from evosax.algorithms import Sep_CMA_ES
        strategy = Sep_CMA_ES(population_size=256, solution=initialization)
      elif args.init_FF_type == 'snes':      
        from evosax.algorithms import SNES
        strategy = SNES(population_size=256, solution=initialization)
      elif args.init_FF_type == 'openes':      
        from evosax.algorithms import Open_ES
        strategy = Open_ES(population_size=256, solution=initialization)
      elif args.init_FF_type == 'pgpe':      
        from evosax.algorithms import PGPE
        strategy = PGPE(population_size=256, solution=initialization)
      elif args.init_FF_type == 'lga':      
        from evosax.algorithms import LGA
        strategy = LGA(population_size=256, solution=initialization)
      elif args.init_FF_type == 'diffev':      
        from evosax.algorithms import DiffusionEvolution
        strategy = DiffusionEvolution(population_size=256, solution=initialization)

      es_params = strategy.default_params
      state = strategy.init(rng, initialization, es_params)

      #state = state.replace(best_member=jnp.array(initialization))
      #state = state.replace(mean=jnp.array(initialization))
      print("[INFO] Init Params", initialization)
      print("[INFO] Starting loss", new_loss_func(jnp.array(initialization), *args_loss))
      #l_f = jax.vmap(new_loss_func, in_axes=(0,None,None,None,None,None,None,None,None,None,None))
      #                               out_axes=(0,None,None,None,None,None,None,None,None,None,None))

      # Run ask-eval-tell loop - NOTE: By default minimization!
      gen_start = time.time()
      fit_list = []
      for t in range(args.random_sample_count):
        #TODO: include vmap
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_gen, state, es_params)
        x = jnp.clip(x, bounds[:,0], bounds[:,1])
        fitness = jnp.array([new_loss_func(p, *args_loss) for p in x], dtype=jnp.float32)
        state, metrics = strategy.tell(rng_eval, x, fitness, state, es_params)
        if (t + 1) % 10 == 0:
          print("# Gen: {}|Fitness: {:.5f}".format(t+1, state.best_fitness))

      print("[INFO] Best Solution:", state.best_solution)
      print("[INFO] Best Fitness:", state.best_fitness)
      selected_params = state.best_solution
      gen_end = time.time()
      print("Genetic Optimization Time:", gen_end-gen_start)

    if args.init_FF_type == 'pcmaes':
      sys.exit("[ERROR] Parallel GAs not fully implemented")
      #TODO example implementation for
      #parallel solver using shard map
      from jax.sharding import Mesh
      from jax.sharding import PartitionSpec
      from jax.sharding import NamedSharding
      from jax.experimental import mesh_utils
      from jax.experimental.shard_map import shard_map

      print("JAX Devices", jax.devices())
      P = jax.sharding.PartitionSpec
      devices = mesh_utils.create_device_mesh((4,))
      mesh = jax.sharding.Mesh(devices, ('x'))
      sharding = jax.sharding.NamedSharding(mesh, P('x'))

      rng = jax.random.PRNGKey(0)
      rng = jax.random.split(rng, 4)
      args_loss = (param_indices, force_field, training_data,
          list_positions, aligned_data, center_sizes, False,
          aligned_amber_ff, ff_type_int, charge_type_int)
      es_params = strategy.default_params
      es_params = es_params.replace(mu_eff=jnp.repeat(es_params.mu_eff, 4))
      es_params = es_params.replace(c_1=jnp.repeat(es_params.c_1, 4))
      es_params = es_params.replace(c_mu=jnp.repeat(es_params.c_mu, 4))
      es_params = es_params.replace(c_sigma=jnp.repeat(es_params.c_sigma, 4))
      es_params = es_params.replace(d_sigma=jnp.repeat(es_params.d_sigma, 4))
      es_params = es_params.replace(c_c=jnp.repeat(es_params.c_c, 4))
      es_params = es_params.replace(chi_n=jnp.repeat(es_params.chi_n, 4))
      es_params = es_params.replace(c_m=jnp.repeat(es_params.c_m, 4))
      es_params = es_params.replace(sigma_init=jnp.repeat(es_params.sigma_init, 4))

      es_params = es_params.replace(init_min=jnp.broadcast_to(bounds[:,0],(4,)+bounds[:,0].shape))
      es_params = es_params.replace(init_max=jnp.broadcast_to(bounds[:,1],(4,)+bounds[:,1].shape))
      es_params = es_params.replace(clip_min=jnp.broadcast_to(bounds[:,0],(4,)+bounds[:,0].shape))
      es_params = es_params.replace(clip_max=jnp.broadcast_to(bounds[:,1],(4,)+bounds[:,1].shape))

      #TODO probably a better way to do this by iterating over the fields and doing replace(**kwargs)
      #for field in es_params.fields

      #init_fn = shard_map(strategy.initialize, mesh=mesh, in_specs=(P(None), P("x")), out_specs=P("x"))
      init_fn = jax.jit(jax.vmap(strategy.initialize))
      print("clip shape", es_params.clip_max.shape)
      print("mu shape", es_params.c_mu.shape)
      es_params_sharded = jax.device_put(es_params, sharding)
      rng_sharded = jax.device_put(rng, sharding)

      #jax.debug.visualize_array_sharding(es_params_sharded)
      #print("es sharded devices", es_params_sharded.init_min.devices())

      state = init_fn(rng_sharded, es_params_sharded)
      #state = init_fn(rng, es_params)

      print("PCMAES Best Member", state.best_member.shape)

      gen_start = time.time()

      #TODO: better to do jit of shard map than shard map of jit if i go that direction
      @jax.jit
      @jax.vmap
      def opt_loop(rng, es_params, state):
        jax.debug.print("Optimization loop starting")
        for t in range(args.generations):
          #TODO: include vmap
          rng, rng_gen, rng_eval = jax.random.split(rng, 3)
          x, state = strategy.ask(rng_gen, state, es_params)
          fitness = jnp.array([new_loss_func(p, *args_loss) for p in x], dtype=jnp.float32)
          state = strategy.tell(x, fitness, state, es_params)
          if (t + 1) % 1 == 0:
            jax.debug.print("# Gen: {gen}", gen=t+1)
            sys.stdout.flush()

        return state

      state = opt_loop(rng_sharded, es_params_sharded, state)

      print("Best Member:", state.best_member)
      print("Best Fitness:", state.best_fitness)
      selected_params = state.best_member
      gen_end = time.time()
      print("Genetic Optimization Time:", gen_end-gen_start)

      # TODO try to gather at end with lax.all_gather?
      sys.exit()

    ##################################################################################################
  
    [global_min_params,
     global_min,
     center_sizes] = train_FF(selected_params, param_indices, bounds, force_field,
                           aligned_data, center_sizes, training_data,
                           validation_data,
                           num_steps, e_minim_flag, opt_method, optim_options,
                           advanced_opts,
                           new_loss_and_grad_func, minim_func, allocate_func, args.ff_type, args.opt_mode, ffq_ff)
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
    if args.ff_type == "reaxff":
      force_field = set_params_jit(force_field, param_indices, params, args.ff_type, args.opt_mode)
    elif args.ff_type == "ambereem":
      ffq_ff = set_params_jit(ffq_ff, param_indices, params, args.ff_type, args.opt_mode)
    elif args.ff_type == "amber":
      # TODO decide how to dump params
      # TODO make this a subset of an output writer module that can call
      # the right function based on the ff mode
      # could use parmed or just dump labeled json
      # also need to work out .dat input for these
      print("final amber params", params)
      print("param indices", param_indices)
      sys.exit("[ERROR] Amber output not finished")
      force_field = set_params_jit(force_field, param_indices, params, args.ff_type, args.opt_mode)
    if e_minim_flag:
      minim_start = time.time()
      [list_positions, cur_total_energy,
      center_sizes, cur_RMSG_vals] = minim_func(aligned_data,
                                                center_sizes,
                                                force_field,
                                                ffq_ff=ffq_ff,
                                                ff_type=args.ff_type)
      minim_end = time.time()
    else:
      if args.ff_type == "reaxff":
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
                                      center_sizes, args.ff_type, args.opt_mode, ffq_ff,
                                      True)
    for k in indiv_errors.keys():
      # move data to regular numpy arrays
      for i,sub_val in enumerate(indiv_errors[k]):
        indiv_errors[k][i] = onp.array(sub_val)
    loss = float(loss)
    loss_str = str(round(loss))
    new_name = "{}/new_FF_{}_{}".format(args.out_folder,unique_id,loss_str)
    if args.ff_type == "reaxff":
      new_force_field = move_dataclass(force_field, onp)
      parse_and_save_force_field(args.init_FF, new_name, new_force_field)
    elif args.ff_type == "ambereem":
      ffq_name = "{}/ffq_params_{}_{}.dat".format(args.out_folder,unique_id,loss_str)
      new_force_field = move_dataclass(ffq_ff, onp)
      parse_and_save_force_field_amber(f_list, new_force_field, args.ffq_params, ffq_name)
    elif args.ff_type == "amber":
      sys.exit("[ERROR] Amber output not finished")
  
    # TODO add mean and stdev along with any other relevant statistics
    report_name = "{}/report_{}_{}.txt".format(args.out_folder,unique_id,loss_str)
    produce_error_report(report_name, training_data, indiv_errors, geo_index_to_name)
  
    # produce the report for the validation data if available
    if args.use_valid:
      [valid_loss, 
       valid_indiv_errors] = new_loss_func(params, param_indices,
                                        force_field, validation_data,
                                        list_positions, aligned_data,
                                        center_sizes, args.ff_type, args.opt_mode,
                                        True)
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
