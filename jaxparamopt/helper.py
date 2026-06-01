"""
Contains helper functions for I/O and training

Author: Mehmet Cagri Kaymak
"""

import  os
import jax
import jax.numpy as jnp
import numpy as onp
import time
import sys
from multiprocessing import get_context
from tabulate import tabulate
import math
import copy
from jaxparamopt.clustering import modified_kmeans
from jaxparamopt.trainingdata import ChargeItem, EnergyItem, DistItem, AngleItem 
from jaxparamopt.trainingdata import TorsionItem, ForceItem, RMSGItem, HessianItem, TrainingData
from jaxparamopt.structure import Structure, BondRestraint, AngleRestraint, TorsionRestraint
from jaxparamopt.inter_list_counter import pool_handler_for_inter_list_count
from jax_md import dataclasses
from jax_md.mm_forcefields.reaxff.reaxff_forcefield import ForceField
import argparse
#from jax_md.amber.amber_helper import GAFFTYPES
import h5py
from collections import defaultdict

# Since we shouldnt access the private API (jaxlib), create a dummy jax array
# and get the type information from the array.
# from jaxlib.xla_extension import ArrayImpl as JaxArrayType
# TODO it's wise to avoid this in situations where another subprocess may call this in a CPU only code
# it can cause issues with invalid GPU context information being copied to subprocess
# JaxArrayType = type(jnp.zeros(1))

def build_float_range_checker(min_v, max_v):
  '''
  Returns a function that can be used to validate fiven FP value
  withing the allowed range ([min_v, max_v])
  '''
  def range_checker(arg):
    try:
      val = float(arg)
    except ValueError:    
      raise argparse.ArgumentTypeError("Value must be a floating point number")
    if val < min_v or val > max_v:
      raise argparse.ArgumentTypeError("Value must be in range [" + str(min_v) + ", " + str(max_v)+"]")
    return val
  return range_checker

def apply_params_field(base_arr, raveled, src, theta):
  if raveled.size == 0:
      return base_arr
  updates = theta[src] # [T]
  flat = base_arr.reshape(-1)
  flat = flat.at[raveled].set(updates) # single fused scatter
  return flat.reshape(base_arr.shape)

def set_params_clusters(ff_clusters, theta, targets):
  # ff_clusters: list/tuple of dataclass ForceField, one per cluster, where each field has leading K_c axis
  new_clusters = []
  for c, ff in enumerate(ff_clusters):
      # for each field present in ff, apply if we have targets
      updates = {}
      for fname in ff.__dataclass_fields__.keys():
          if c in targets and fname in targets[c]:
              r = targets[c][fname]["raveled"]
              s = targets[c][fname]["src"]
              arr = getattr(ff, fname)
              updates[fname] = apply_params_field(arr, r, s, theta)
          else:
              updates[fname] = getattr(ff, fname)
      new_clusters.append(dataclasses.replace(ff, **updates))
  return new_clusters

# unneeded with loss f_n wrapper but may have some use elsewhere
# def segment_sum_field(grad_arr, raveled, src, n_theta):
#   if raveled.size == 0:
#       return jnp.zeros((n_theta,), grad_arr.dtype)
#   g_slots = grad_arr.reshape(-1)[raveled]
#   return jax.ops.segment_sum(g_slots, src, n_theta)

# def grads_to_theta(grad_ff_clusters, targets, n_theta):
#   g_theta = jnp.zeros((n_theta,), grad_ff_clusters[0].__dict__[
#       next(iter(grad_ff_clusters[0].__dict__.keys()))
#   ].dtype)
#   for c, gff in enumerate(grad_ff_clusters):
#       for fname, arr in gff.__dict__.items():
#           if c in targets and fname in targets[c]:
#               r = targets[c][fname]["raveled"]; s = targets[c][fname]["src"]
#               g_theta = g_theta + segment_sum_field(arr, r, s, n_theta)
#   return g_theta

# alternative version that should work better for clusters
def get_params_clusters(ff_clusters, targets, n_theta):
  """
  Returns a 1D array of length n_theta with parameter values in optimizer order,
  read directly from the (clustered) force-field arrays.

  ff_clusters: tuple/list of per-cluster FF dataclasses (arrays shaped (K_c, ...))
  targets:     per (cluster, field) mapping from build_targets
               tgt[c][fname] = {"raveled": int[t_idxs], "src": int[t_idxs]}
  n_theta:     number of optimizer variables (size of flat 0)
  """
  vals_chunks = []
  src_chunks  = []

  for c, per_field in targets.items():
    ff_c = ff_clusters[c]
    for fname, spec in per_field.items():
      r = spec["raveled"]
      s = spec["src"]
      if r.size == 0:
          continue
      arr = getattr(ff_c, fname) # shape (K_c, *inner)
      vals = arr.reshape(-1)[r] # gather representatives for this (c, field)
      vals_chunks.append(vals)
      src_chunks.append(s)

  if not vals_chunks:
    # unlikely case where no targets are mapped; return zeros of an arbitrary dtype
    # probably a more robust way to do this, set dtype based on precision, etc
    return jnp.zeros((n_theta,), dtype=jnp.float32)

  # trick using segment sum should reduce compilation burden compared to individual
  # index into cluster values, need to test more and also ensure correctness
  vals_all = jnp.concatenate(vals_chunks) # [T_total]
  src_all  = jnp.concatenate(src_chunks) # [T_total], each in [0, n_theta)
  # average across duplicates so that identical entries stay identical
  sum_vals = jax.ops.segment_sum(vals_all,  src_all, n_theta) # [n_theta]
  counts   = jax.ops.segment_sum(jnp.ones_like(vals_all), src_all, n_theta) # [n_theta]
  params   = sum_vals / jnp.maximum(counts, 1)

  return params

def split_dataclass(data):
  '''
  From a dataclass with batched atrributes, seperate each sample
  and create a list of samples
  '''
  result = []
  field_names = [field.name for field in dataclasses.fields(data)]
  val = getattr(data, field_names[0])
  size = len(val)
  for i in range(size):
    sub = filter_dataclass(data, i)
    result.append(sub)
  return result

def filter_dataclass(data, filter_map):
  '''
  Apply a given filter to a dataclass with batched atrributes
  '''
  sel_dict = {}
  field_names = [field.name for field in dataclasses.fields(data)]
  d_class = data.__class__
  for attr in field_names:
    val = getattr(data, attr)
    if type(val) in [type(jnp.zeros(1)), onp.ndarray]:
      sel_dict[attr] = val[filter_map]
    # recursive filtering since dataclass might contain other dataclasses
    # as an attribute
    #TODO: if there is self reference, this will cause stack overflow
    if dataclasses.is_dataclass(val):
      sel_dict[attr] = filter_dataclass(val, filter_map)

  return d_class(**sel_dict)

def move_dataclass(obj, target_numpy):
  '''
  Move a given dataclass object to target numpy (either onp or jnp)
  '''
  field_names = [field.name for field in dataclasses.fields(obj)]
  replace_dict = dict()
  for attr in field_names:
    val = getattr(obj, attr)
    if type(val) in [type(jnp.zeros(1)), onp.ndarray]:
      replace_dict[attr] = target_numpy.array(val)
    #TODO: if there is self reference, this will cause stack overflow
    if dataclasses.is_dataclass(val):
      replace_dict[attr] = move_dataclass(val, target_numpy)
  new_obj = dataclasses.replace(obj, **replace_dict)
  return new_obj


def cluster_systems_for_aligning(size_dicts, num_cuts=5,
                                 max_iterations=100,
                                 rep_count=20, print_mode=True):
  '''
  Cluster the similar structures together for aligning them while
  minimizing padding
  '''
  #run the modified k-means algorithm to form the clusters
  [labels,
   min_centr,
   min_counts,
   min_cost] = modified_kmeans(size_dicts,
                               k=num_cuts,
                               max_iterations=max_iterations,
                               rep_count=rep_count,
                               print_mode=print_mode)

  all_cut_indices = [[] for i in range(num_cuts)]
  for i,s in enumerate(size_dicts):
    label = labels[i]
    all_cut_indices[label].append(i)
  
  centers = []
  for group in all_cut_indices:
    my_center = copy.deepcopy(size_dicts[group[0]])
    for i in group[1:]:
      for k in my_center.keys():
        my_center[k] = max(my_center[k], size_dicts[i][k])
    centers.append(my_center)
        
  return all_cut_indices, min_cost, centers


def count_inter_list_sizes(systems, force_field,
                           num_threads=1, pool=None, chunksize=32,
                           close_cutoff=5.0, far_cutoff=10.0):
  '''
  Calculate the interaction list sizes for given list of structures
  '''
  force_field = move_dataclass(force_field, onp)
  start = time.time()
  # get_context("fork") needed for Mac arm processors
  if pool == None:
    my_pool = get_context("fork").Pool(num_threads)
  else:
    my_pool = pool
  size_dicts = pool_handler_for_inter_list_count(systems, force_field,
                                                 my_pool, chunksize,
                                                 close_cutoff, far_cutoff)
  end = time.time()
  if pool == None:
    my_pool.terminate()
  print("Multithreaded interaction list counting took {:.2f} secs with {} threads".format(end-start,num_threads))
  return size_dicts


def process_and_cluster_geos(systems,force_field,max_num_clusters=10,
                             num_threads=1,chunksize=1,
                             close_cutoff=5.0, far_cutoff=10.0):
  '''
  Calculate the interaction list sizes for given list of structures first
  then cluster the similar structures together
  '''  
  size_dicts = count_inter_list_sizes(systems, force_field, 
                                      num_threads=num_threads, chunksize=chunksize,
                                      close_cutoff=close_cutoff,
                                      far_cutoff=far_cutoff)

  all_costs_old = []
  prev = -1
  selected_n_cut = 0
  for n_cut in range(1,max_num_clusters+1):
    all_cut_indices, cost_total, center_sizes = cluster_systems_for_aligning(size_dicts,num_cuts=n_cut,max_iterations=1000,rep_count=1000,print_mode=False)
    #print("Cost with {} clusters: {}".format(n_cut, cost_total))
    all_costs_old.append(cost_total)
    if prev != -1 and cost_total > prev or (prev-cost_total) / prev < 0.15:
        selected_n_cut = n_cut - 1
        break
    prev = cost_total
  #sys.exit()
  if selected_n_cut == 0:
    selected_n_cut = max_num_clusters
  all_cut_indices, cost_total, center_sizes = cluster_systems_for_aligning(size_dicts,num_cuts=selected_n_cut,max_iterations=1000,rep_count=1000,print_mode=True)

  globally_sorted_indices = []
  for l in all_cut_indices:
    for ind in l:
      globally_sorted_indices.append(ind)
  return globally_sorted_indices, all_cut_indices, center_sizes


def build_energy_report_item(energy_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for energy item
  '''
  sys_inds = energy_item.sys_inds
  multips = energy_item.multip
  out_str = ""
  for i in range(len(sys_inds)):
    if multips[i] == 0.0:
      continue
    name = geo_index_to_name[sys_inds[i]]
    div = round(1.0/multips[i])
    sign_str = "+"
    if div < 0:
      sign_str = "-"
    out_str += f"{sign_str} {name}/{abs(div)} "
  out_str = out_str[:-1]
  out_str = "ENERGY: " + out_str
  row = [out_str, energy_item.weight, energy_item.target, pred, weighted_error]
  return row

def build_force_report_item(force_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for force item
  '''
  name = geo_index_to_name[force_item.sys_ind]
  out_str = f"{name} {force_item.a_ind + 1}"
  dirs = ["X", "Y", "Z"]
  rows = []
  for i in range(3):
    new_out_str = f"FORCE-{dirs[i]}: " + out_str
    row = [new_out_str, force_item.weight, float(force_item.target[i]),
     float(pred[i]), float(weighted_error[i])]
    rows.append(row)
  return rows

def build_hessian_report_item(hessian_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for force item
  '''
  # TODO should this take an entire row of the hessian?
  # there is probably a more compact way to express this
  name = geo_index_to_name[hessian_item.sys_ind]
  out_str = f"{name} {hessian_item.a_ind + 1}"
  rows = []
  num_columns = len(hessian_item)
  for i in range(num_columns):
    new_out_str = f"HESSIAN-{i}: " + out_str
    row = [new_out_str, hessian_item.weight, float(hessian_item.target[i]),
     float(pred[i]), float(weighted_error[i])] # TODO sum weighted error?
    rows.append(row)
  return rows

def build_charge_report_item(charge_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for charge item
  '''
  name = geo_index_to_name[charge_item.sys_ind]
  out_str = f"{name} {charge_item.a_ind + 1}"
  out_str = "CHARGE: " + out_str
  row = [out_str, charge_item.weight, charge_item.target, pred, weighted_error]
  return row

def build_distance_report_item(distance_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for distance item
  '''
  name = geo_index_to_name[distance_item.sys_ind]
  out_str = f"{name} {distance_item.a1_ind + 1} {distance_item.a2_ind + 1}"
  out_str = "DISTANCE: " + out_str
  row = [out_str, distance_item.weight, distance_item.target, pred, weighted_error]
  return row

def build_angle_report_item(angle_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report row for angle item
  '''
  name = geo_index_to_name[angle_item.sys_ind]
  out_str = f"{name} {angle_item.a1_ind + 1} {angle_item.a2_ind + 1} {angle_item.a3_ind + 1}"
  out_str = "ANGLE: " + out_str
  row = [out_str, angle_item.weight, angle_item.target, pred, weighted_error]
  return row

def build_torsion_report_item(torsion_item, pred, weighted_error, geo_index_to_name):
  '''
  Build the report for torsion items
  '''
  name = geo_index_to_name[torsion_item.sys_ind]
  out_str = f"{name} {torsion_item.a1_ind + 1} {torsion_item.a2_ind + 1} {torsion_item.a3_ind + 1} {torsion_item.a4_ind + 1}"
  out_str = "TORSION: " + out_str
  row = [out_str, torsion_item.weight, torsion_item.target, pred, weighted_error]
  return row

# Produces a report with item based error (similar to what the standalone code does)
def produce_error_report(filename, tranining_items, indiv_error, geo_index_to_name):
  '''
  Produce an error report, similar to how the standalone code does it
  '''
  fptr = open(filename, 'w')
  headers = ["Item Text", "Weight", "Target", "Prediction", "Weighted Error", "Cum. Sum."]
  data_to_print = []
  cumulative_err = 0.0
  functions = {"ENERGY":build_energy_report_item,
               "CHARGE":build_charge_report_item,
               "FORCE":build_force_report_item,
               "HESSIAN":build_hessian_report_item,
               "DISTANCE":build_distance_report_item,
               "ANGLE":build_angle_report_item,
               "TORSION":build_torsion_report_item}

  attributes = {"ENERGY":"energy_items",
               "CHARGE":"charge_items",
               "FORCE":"force_items",
               "HESSIAN":"hessian_items",
               "DISTANCE":"dist_items",
               "ANGLE":"angle_items",
               "TORSION":"torsion_items"}

  for key, attr in attributes.items():
    if key not in indiv_error:
      continue
    sub_items = getattr(tranining_items, attr)
    sub_items = move_dataclass(sub_items, onp)
    sub_items = split_dataclass(sub_items)
    [preds, targets, weighted_errors] = indiv_error[key]
    for i, item in enumerate(sub_items): 
      row = functions[key](item, preds[i], weighted_errors[i], geo_index_to_name)
      if key == "FORCE":
        rows = row
        for j, row in enumerate(rows):
          cumulative_err += float(weighted_errors[i][j])
          row.append(cumulative_err)
          data_to_print.append(row)
      else:
        cumulative_err += weighted_errors[i]
        row.append(cumulative_err)
        data_to_print.append(row)


  table = tabulate(data_to_print, headers, floatfmt=".2f")
  print(table, file=fptr)
  fptr.close()


# with the current structure for single/multi/group params
#  - ("single", ('sigma', (i,)))
#  - ("multi", ff_idx, ('sigma', (i,)))
#  - ("group", ((('sigma',(i0,)), ff_idx0), (('sigma',(i1,)), ff_idx1), ...))
#
# and given:
#  - all_cut_indices: List[List[int]]  # e.g., [[0,1,4], [2,3,5]] for 2 clusters
#  - field_shapes: Dict[field_name, Dict[cluster_id, Tuple[int,...]]]  # inner shapes for alignment

def global_to_cluster_ff(all_cut_indices):
  # returns dict: global_ff_idx -> (cluster_id, local_ff_idx)
  lut = {}
  for c, ff_ids in enumerate(all_cut_indices):
    for local, g in enumerate(ff_ids):
      lut[g] = (c, local)
  return lut

def ravel_for_cluster(cluster_k, inner_shape, local_ff_idx, inner_idx):
  # inner_idx is a tuple like (i,) or (i,j, ...)
  # shape = (cluster_k,) + inner_shape
  shape = inner_shape
  multi = (local_ff_idx,) + inner_idx
  return jnp.ravel_multi_index(jnp.array(multi), jnp.array(shape)).item()

def build_targets(params_list, all_cut_indices, force_field):
  g2c = global_to_cluster_ff(all_cut_indices)

  # TODO some of this probably isn't needed after making the changes to get and
  # set params, probably just need the cluster targetes
  # targets[cluster_id][field_name] -> dict with python lists
  targets = defaultdict(lambda: defaultdict(lambda: {"raveled": [], "src": []}))
  # for reporting current values
  src_single, src_multi, src_group_rep = [], [], []

  theta_src = 0  # walk through optimizer vector positions
  for entry in params_list:
    tag = entry[0]

    if tag == "single":
      field_name, inner_idx = entry[1] # ('sigma', (i,))
      # single implies one FF (global) -> treat as "multi" with chosen FF (here 0th or fixed)
      # in single-FF run, there's only one FF object -> it belongs to exactly one cluster with K_c==1.
      # if only one ff globally, put its global ff_idx in config; here assume 0
      global_ff_idx = 0
      c, local = g2c[global_ff_idx]
      Kc = len(all_cut_indices[c])
      print("c val", c, local)
      #inner_shape = force_field[c][field_name].shape
      #inner_shape = force_field[field_name].shape
      inner_shape = getattr(force_field[c],field_name).shape
      rav = ravel_for_cluster(Kc, inner_shape, local, inner_idx)
      targets[c][field_name]["raveled"].append(rav)
      targets[c][field_name]["src"].append(theta_src)

      src_single.append(theta_src)
      theta_src += 1

    elif tag == "multi":
      global_ff_idx = entry[1]
      field_name, inner_idx = entry[2]
      c, local = g2c[global_ff_idx]
      Kc = len(all_cut_indices[c])
      #inner_shape = force_field[c][field_name].shape
      inner_shape = getattr(force_field[c],field_name).shape
      rav = ravel_for_cluster(Kc, inner_shape, local, inner_idx)
      targets[c][field_name]["raveled"].append(rav)
      targets[c][field_name]["src"].append(theta_src)

      src_multi.append(theta_src)
      theta_src += 1

    elif tag == "group":
      group_items = entry[1]  # tuple of ((field_name, inner_idx), global_ff_idx)
      # one optimizer source for the whole group:
      group_src = theta_src
      # choose representative for "get params" reporting (e.g., first):
      src_group_rep.append(group_src)
      # push each target in the group
      for ((field_name, inner_idx), global_ff_idx) in group_items:
        c, local = g2c[global_ff_idx]
        Kc = len(all_cut_indices[c])
        # this may not be very efficient, can't it be replaced with max_sizes?
        inner_shape = getattr(force_field[c],field_name).shape
        rav = ravel_for_cluster(Kc, inner_shape, local, inner_idx)
        targets[c][field_name]["raveled"].append(rav)
        targets[c][field_name]["src"].append(group_src)
      theta_src += 1

    else:
      raise ValueError(f"Unknown param tag: {tag}")

  # convert python lists -> jnp arrays (int32) for each (cluster, field)
  tgt = {}
  for c, fields in targets.items():
    tgt[c] = {}
    for fname, d in fields.items():
      r = jnp.array(d["raveled"], dtype=jnp.int32)
      s = jnp.array(d["src"],     dtype=jnp.int32)
      tgt[c][fname] = {"raveled": r, "src": s}

  report_src = {
    "single": jnp.array(src_single, dtype=jnp.int32),
    "multi":  jnp.array(src_multi,  dtype=jnp.int32),
    "group":  jnp.array(src_group_rep, dtype=jnp.int32),
  }
  n_theta = theta_src
  return tgt, report_src, n_theta


def parse_and_save_force_field(old_ff_file, new_ff_file,force_field):
  '''
  Save the force field to a file
  '''
  output = ""
  f = open(old_ff_file, 'r')
  line = f.readline()
  output = output + line
  header = line.strip()

  line = f.readline()
  output = output + line
  num_params = int(line.strip().split()[0])
  global_params = jnp.zeros(shape=(num_params,1), dtype=jnp.float64)
  ff = force_field
  for i in range(num_params):
    line = f.readline()
    line = list(line)
    #-------------------------------------------------------------
    if i == 0:
      line[:10] = "{:10.4f}".format(ff.over_coord1[0])  #overcoord1
    if i == 1:
      line[:10] = "{:10.4f}".format(ff.over_coord2[0]) #overcoord2
    #-------------------------------------------------------------

    #-------------------------------------------------------------
    if i == 3:
      line[:10] = "{:10.4f}".format(ff.trip_stab4[0])  #trip_stab4
    if i == 4:
      line[:10] = "{:10.4f}".format(ff.trip_stab5[0]) #trip_stab5
    if i == 7:
      line[:10] = "{:10.4f}".format(ff.trip_stab8[0])  #trip_stab8
    if i == 10:
      line[:10] = "{:10.4f}".format(ff.trip_stab11[0]) #trip_stab11
    #-------------------------------------------------------------
    #valency related parameters
    if i == 2:
      line[:10] = "{:10.4f}".format(ff.val_par3[0])  #val_par3
    if i == 14:
      line[:10] = "{:10.4f}".format(ff.val_par15[0]) #val_par15
    if i == 15:
      line[:10] = "{:10.4f}".format(ff.par_16[0])  #par_16
    if i == 16:
      line[:10] = "{:10.4f}".format(ff.val_par17[0]) #val_par17
    if i == 17:
      line[:10] = "{:10.4f}".format(ff.val_par18[0])  #val_par18
    if i == 19:
      line[:10] = "{:10.4f}".format(ff.val_par20[0]) #val_par20
    if i == 20:
      line[:10] = "{:10.4f}".format(ff.val_par21[0])  #val_par21
    if i == 30:
      line[:10] = "{:10.4f}".format(ff.val_par31[0]) #val_par31
    if i == 33:
      line[:10] = "{:10.4f}".format(ff.val_par34[0])  #val_par34
    if i == 38:
      line[:10] = "{:10.4f}".format(ff.val_par39[0]) #val_par39
    #-------------------------------------------------------------

    #-------------------------------------------------------------
    #over-under coord.
    if i == 5:
      line[:10] = "{:10.4f}".format(ff.par_6[0])  #par_6
    if i == 6:
      line[:10] = "{:10.4f}".format(ff.par_7[0]) #par_7
    if i == 8:
      line[:10] = "{:10.4f}".format(ff.par_9[0])  #par_9
    if i == 9:
      line[:10] = "{:10.4f}".format(ff.par_10[0]) #par_10
    if i == 31:
      line[:10] = "{:10.4f}".format(ff.par_32[0])  #par_32
    if i == 32:
      line[:10] = "{:10.4f}".format(ff.par_33[0]) #par_33

    #-------------------------------------------------------------
    #torsion
    if i == 23:
      line[:10] = "{:10.4f}".format(ff.par_24[0])  #par_24
    if i == 24:
      line[:10] = "{:10.4f}".format(ff.par_25[0]) #par_25
    if i == 25:
      line[:10] = "{:10.4f}".format(ff.par_26[0])  #par_26
    if i == 27:
      line[:10] = "{:10.4f}".format(ff.par_28[0]) #par_28

    #-------------------------------------------------------------
    # vdw
    if i == 28:
      line[:10] = "{:10.4f}".format(ff.vdw_shiedling[0]) #vdw_shiedling
    output = output + ''.join(line)

  line = f.readline()
  output = output + line

  num_atom_types = int(line.strip().split()[0])
  # skip 3 lines of comment
  output = output + f.readline()
  output = output + f.readline()
  output = output + f.readline()

  atom_names = []
  line_ctr = 0
  for i in range(num_atom_types):
    # first line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.rat[i]) #rat - rob1
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.rvdw[i]) #rvdw
    line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.eps[i]) #eps
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.gamma[i]) #gamma
    line[3 + 9 * 6:3 + 9 * 7] = "{:9.4f}".format(ff.rapt[i]) #rapt - rob2
    line[3 + 9 * 7:3 + 9 * 8] = "{:9.4f}".format(ff.stlp[i]) #stlp

    output = output + ''.join(line)

    # second line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.alf[i]) #alf
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vop[i]) #vop
    line[3 + 9 * 2:3 + 9 * 3] = "{:9.4f}".format(ff.valf[i]) #valf
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.valp1[i]) #valp1
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.electronegativity[i])
    line[3 + 9 * 6:3 + 9 * 7] = "{:9.4f}".format(ff.idempotential[i])

    output = output + ''.join(line)
    # third line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.vnq[i]) #vnq - rob3
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vlp1[i]) #vlp1
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.bo131[i]) #bo131
    line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.bo132[i]) #bo132
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.bo133[i]) #bo133

    output = output + ''.join(line)

    # fourth line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.vovun[i])
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vval1[i])
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.vval3[i])
    line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.vval4[i])

    output = output + ''.join(line)



  line = f.readline()  # num_bonds
  output = output + line

  line = line.strip()
  num_bonds = int(line.split()[0])
  output = output + f.readline() # skip next line (comment)
  for _ in range(num_bonds):
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i = int(tmp[0]) - 1 # index starts at 0
    j = int(tmp[1]) - 1

    line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.de1[i,j])
    line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.de2[i,j])
    line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.de3[i,j])
    line[6 + 9 * 3:6 + 9 * 4] = "{:9.4f}".format(ff.psi[i,j])
    line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.pdo[i,j])
    line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.v13cor[i,j])
    line[6 + 9 * 6:6 + 9 * 7] = "{:9.4f}".format(ff.popi[i,j])
    line[6 + 9 * 7:6 + 9 * 8] = "{:9.4f}".format(ff.vover[i,j])
    #print(''.join(line))
    output = output + ''.join(line)
    # second line
    line = f.readline()
    line = list(line)

    line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.psp[i,j])
    line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.pdp[i,j])
    line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.ptp[i,j])
    line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.bop1[i,j])
    line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.bop2[i,j])
    line[6 + 9 * 6:6 + 9 * 7] = "{:9.4f}".format(ff.ovc[i,j])
    #print(''.join(line))
    output = output + ''.join(line)
  line = f.readline()  # num_off_diag
  output = output + line

  line = line.strip()
  num_off_diag = int(line.split()[0])

  for _ in range(num_off_diag):
    # first line
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i = int(tmp[0]) - 1 # index starts at 0
    j = int(tmp[1]) - 1

    line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.p2co_off[i,j])
    line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.p1co_off[i,j])  # was /2
    line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.p3co_off[i,j])
    line[6 + 9 * 3:6 + 9 * 4] = "{:9.4f}".format(ff.rob1_off[i,j])
    line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.rob2_off[i,j])
    line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.rob3_off[i,j])

    output = output + ''.join(line)

  #valency angle parameters
  line = f.readline()  # num_val_params
  output = output + line

  line = line.strip()
  num_val_params = int(line.split()[0])

  for _ in range(num_val_params):
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i = int(tmp[0]) - 1 # index starts at 0
    j = int(tmp[1]) - 1
    k = int(tmp[2]) - 1

    line[9 + 9 * 0:9 + 9 * 1] = "{:9.4f}".format(ff.th0[i,j,k])
    line[9 + 9 * 1:9 + 9 * 2] = "{:9.4f}".format(ff.vka[i,j,k])
    line[9 + 9 * 2:9 + 9 * 3] = "{:9.4f}".format(ff.vka3[i,j,k])
    line[9 + 9 * 3:9 + 9 * 4] = "{:9.4f}".format(ff.vka8[i,j,k])
    line[9 + 9 * 4:9 + 9 * 5] = "{:9.4f}".format(ff.vkac[i,j,k])
    line[9 + 9 * 5:9 + 9 * 6] = "{:9.4f}".format(ff.vkap[i,j,k])
    line[9 + 9 * 6:9 + 9 * 7] = "{:9.4f}".format(ff.vval2[i,j,k])

    output = output + ''.join(line)

  #torsion parameters
  line = f.readline()  # num_tors_params
  output = output + line

  line = line.strip()
  num_tors_params = int(line.split()[0])

  for _ in range(num_tors_params):
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i1 = int(tmp[0]) - 1 # index starts at 0
    i2 = int(tmp[1]) - 1
    i3 = int(tmp[2]) - 1
    i4 = int(tmp[3]) - 1

    if i1 != -1 and i4 != -1:
      line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[i1,i2,i3,i4])
      line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[i1,i2,i3,i4])
      line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[i1,i2,i3,i4])
      line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[i1,i2,i3,i4])
      line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[i1,i2,i3,i4])

    if i1 == -1 and i4 == -1:
      sel_ind = force_field.num_atom_types - 1
      line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[sel_ind,
                                                              i2,
                                                              i3,
                                                              sel_ind])
    output = output + ''.join(line)

  # hbond parameters
  #torsion parameters
  line = f.readline()  # num_tors_params
  output = output + line

  line = line.strip()
  num_hbond_params = int(line.split()[0])

  for i in range(num_hbond_params):
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i1 = int(tmp[0]) - 1
    i2 = int(tmp[1]) - 1
    i3 = int(tmp[2]) -1
    line[9 + 9 * 0:9 + 9 * 1] = "{:9.4f}".format(ff.rhb[i1,i2,i3])
    line[9 + 9 * 1:9 + 9 * 2] = "{:9.4f}".format(ff.dehb[i1,i2,i3])
    line[9 + 9 * 2:9 + 9 * 3] = "{:9.4f}".format(ff.vhb1[i1,i2,i3])
    line[9 + 9 * 3:9 + 9 * 4] = "{:9.4f}".format(ff.vhb2[i1,i2,i3])
    output = output + ''.join(line)



  # need to append some extra lines because of 0 values
  for line in f:
    output = output + line

  file_new = open(new_ff_file,"w")
  file_new.write(output)
  file_new.close()

  f.close()


