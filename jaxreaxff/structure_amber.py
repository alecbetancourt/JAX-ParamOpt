import jax
import jax.numpy as jnp
import numpy as onp
from jax_md.dataclasses import fields #TODO make sure any invocations of this are from the jmd version, not regular python

from jax_md.amber.amber_helper import load_amber_ff, load_ffq_ff, GAFFTYPES
from jax_md.amber.amber_forcefield import AmberForceField
from jaxreaxff.helper_prmtop import build_prm_list
from jaxreaxff.structure import align_structures, align_and_batch_structures
from concurrent.futures import ThreadPoolExecutor
import time
import os
import json

def load_amber_ff_v2(geo_file, prm_file, pme_flag, charge_model, dtype):
    if pme_flag:
        nonbonded_method = "PME"
    else:
        nonbonded_method = "NoCutoff"
    
    dr_threshold = 0.0 # TODO should this be a toggle, or just go with a safe value like 0.2

    if os.path.isdir(prm_file):
        print("[INFO] --init_FF is a directory, searching for all .prmtop files")

        f_list = build_prm_list(geo_file, prm_file)
        force_field, ffq_ff = load_amber_ff_batch(f_list, None, "amber", dtype=dtype)
    else:
        force_field = load_amber_ff(inpcrd_file=None, prmtop_file=prm_file, 
                        ffq_file=None, nonbonded_method=nonbonded_method,
                        charge_method=charge_model, dr_threshold=dr_threshold, dtype=dtype)
    
    return force_field

# Load force field files and return list of structures
def load_amber_ff_batch(prm_list, ffq_file, ff_type, dtype):
    if ff_type == "amber":
        charge_method = "GAFF" # TODO wrap this and the other types into an enum
    elif ff_type == "ambereem":
        charge_method = "FFQ"

    # TODO placeholders, will have to change for periodic case,
    # might be able to still get box vectors from prmtop
    dr_threshold = 0.0
    nonbonded_method = "NoCutoff"
    inpcrd_file = None

    list_ffs = []

    ##########################################################################

    ### Serial parameter loading code

    # load_start = time.time()
    # for prm_file in prm_list:
    #     print("Loading:", prm_file)
    #     amber_ff = load_amber_ff(inpcrd_file=inpcrd_file, prmtop_file=prm_file, 
    #                     ffq_file=ffq_file, nonbonded_method=nonbonded_method,
    #                     charge_method=charge_method, dr_threshold=dr_threshold, dtype=dtype)
        
    #     list_ffs.append(amber_ff)
    # load_end = time.time()
    # print("Total time to load AMBER FF files", load_end-load_start)

    ###########################################################################

    ### Thread parallel parameter loading code

    # TODO move this outside and add error handling message
    # something to the effect of if slurm not enabled but dlfind is, throw error
    # the likely workflow for this eventualy should be either adding an explicit
    # option for SLURM, figuring out a more agnostic approach to setting this up
    # or implicitly looking for SLURM process mapping and then falling back to
    # looking at os/sys information
    num_tasks = int(os.environ.get("SLURM_NTASKS", "1"))

    load_start = time.time()
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [
        executor.submit(load_amber_ff, inpcrd_file=inpcrd_file, prmtop_file=prm_file, 
                        ffq_file=ffq_file, nonbonded_method=nonbonded_method,
                        charge_method=charge_method, dr_threshold=dr_threshold, dtype=dtype)
        for prm_file in prm_list
        ]
        list_ffs = [future.result() for future in futures]
    load_end = time.time()
    print("Total time to load AMBER FF files (w/ Thread Pool)", load_end-load_start)

    ###########################################################################

    ffq_ff = None
    if ffq_file != None:
        ffq_ff = load_ffq_ff(ffq_file, dtype=dtype)

    return list_ffs, ffq_ff

def map_params_amber(params, mode='group'):
    '''
    Map the read parameters to new type of indexing to select them from
    a given force field object
    '''

    # TODO clean this up and also implement logic to differentiate single vs group case
    # in particular, indexing for things like torsions is going to be more difficult
    # TODO move this to the force field definitions or somewhere more extensible
    # could probably just extract this from params to index as well
    # reax version takes an argument for this, need to look into that more as well
    type_dict = {(2,0):"torsion_k",(2,1):"torsion_phase",(2,2):"torsion_period",(4,0):"scee_14",(4,1):"scnb_14",
                (5,0):"gamma", (5,1):"electronegativity", (5,2):"hardness"}
    # TODO there's a more complex issue underlying all of this
    # parameters like periodicity are only well defined for integers
    # while all of the standard optimizers assume continuous values and gradients
    # i think the original code got around this because parmed will seemingly floor
    # all periodicity values before updating the ff. The issue here being that the code
    # doesn't necessarily do this now and i think it makes the optimization more difficult
    # my thoughts for solutions here are to either add a parameter to the set_params fn to
    # explicitly floor parameters that are marked as integers or set up some pre-optimization scheme
    # in the global section that does brute force search here and then removes those values from the final
    # optimization routine
    new_params = []
    for p in params:
        key = (p[0],p[1],p[2])
        #value = (type_dict[p[2]], (p[1],))
        value = (type_dict[(p[0], p[2])], (p[1],))
        new_item = (value, p[3],p[4],p[5])
        new_params.append(new_item)
    return new_params

def process_and_cluster_geos_amber(systems, batch_size, dtype):
    full_size = len(systems)
    center_sizes = []
    cut_indices = []
    i = 0

    # TODO basic approach for batching geometries, eventually needs to be
    # integrated into k-means with force field structure information
    for bs in range(0,full_size,batch_size):
        max_sizes = {'num_atoms': 0,
            'periodic_image_count': 0}
        cut_idx = []
        for struct in systems[bs:bs+batch_size]:
                cut_idx.append(i)
                i += 1
                atom_mask = struct.atom_types != -1
                max_sizes['num_atoms'] = max(max_sizes['num_atoms'], len(atom_mask))
                max_sizes['periodic_image_count'] = max(max_sizes['periodic_image_count'], len(struct.periodic_image_shifts))
        center_sizes.append(max_sizes)
        cut_indices.append(cut_idx)
    
    return cut_indices, center_sizes

def process_and_cluster_ff_amber(force_fields, batch_size, dtype):
    full_size = len(force_fields)
    center_sizes = []
    cut_indices = []
    i = 0

    for bs in range(0,full_size,batch_size):
        max_sizes = {}
        cut_idx = []
        for ff in force_fields[bs:bs+batch_size]:
            cut_idx.append(i)
            i += 1
            for field in fields(ff):
                old_max_size = max_sizes[field.name] if field.name in max_sizes else -1
                attr = getattr(ff, field.name)
                if isinstance(attr, jnp.ndarray) or isinstance(attr, onp.ndarray):
                    # TODO consider the following, and look at doc page for this because of different behavior
                    # if jnp.isscalar(attr)
                    new_size = 1 if attr.ndim == 0 else len(attr) # for singleton members, should remove this and replace it with the option below
                elif jnp.isscalar(attr) or onp.isscalar(attr):
                    new_size = 1
                else:
                    new_size = -1 # for non array types or scalar types that can't be stacked

                max_sizes[field.name] = max(new_size, old_max_size)
        center_sizes.append(max_sizes)
        cut_indices.append(cut_idx)

    return cut_indices, center_sizes

def align_ff_amber(force_fields, max_sizes, dtype):
    full_size = len(force_fields)

    name = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    atom_types = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    atomic_number = onp.zeros(shape=(full_size, max_sizes["atomic_number"]), dtype=dtype)
    total_charge = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    #params_to_indices = [None] * full_size
    params_to_indices = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    bond_restraints = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    angle_restraints = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    torsion_restraints = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    atom_count = onp.zeros(shape=(full_size), dtype=onp.int32)
    positions = onp.zeros(shape=(full_size, max_sizes["atom_count"], 3), dtype=dtype)
    box_vectors = onp.zeros(shape=(full_size, 3), dtype=dtype)
    masses = onp.zeros(shape=(full_size, max_sizes["masses"]), dtype=dtype)
    cutoff = onp.zeros(shape=(full_size,), dtype=dtype)
    nbr_list = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    grid_points = onp.zeros(shape=(full_size, 3), dtype=dtype)
    ewald_alpha = onp.zeros(shape=(full_size,), dtype=dtype)
    ewald_error = onp.zeros(shape=(full_size,), dtype=dtype)
    dr_threshold = onp.zeros(shape=(full_size,), dtype=dtype)
    exclusions = onp.zeros(shape=(full_size, max_sizes["exclusions"], 2), dtype=onp.int32)-1
    bond_idx = onp.zeros(shape=(full_size, max_sizes["bond_idx"], 2), dtype=onp.int32)-1
    bond_k = onp.zeros(shape=(full_size, max_sizes["bond_k"]), dtype=dtype)
    bond_len = onp.zeros(shape=(full_size, max_sizes["bond_len"]), dtype=dtype)
    angle_idx = onp.zeros(shape=(full_size, max_sizes["angle_idx"], 3), dtype=onp.int32)-1
    angle_k = onp.zeros(shape=(full_size, max_sizes["angle_k"]), dtype=dtype)
    angle_equil = onp.zeros(shape=(full_size, max_sizes["angle_equil"]), dtype=dtype)
    torsion_idx = onp.zeros(shape=(full_size, max_sizes["torsion_idx"], 4), dtype=onp.int32)-1
    torsion_k = onp.zeros(shape=(full_size, max_sizes["torsion_k"]), dtype=dtype)
    torsion_phase = onp.zeros(shape=(full_size, max_sizes["torsion_phase"]), dtype=dtype)
    torsion_period = onp.zeros(shape=(full_size, max_sizes["torsion_period"]), dtype=dtype)
    pairs = onp.zeros(shape=(full_size, max_sizes["pairs"], 2), dtype=onp.int32)-1
    sigma = onp.zeros(shape=(full_size, max_sizes["sigma"]), dtype=dtype)
    epsilon = onp.zeros(shape=(full_size, max_sizes["epsilon"]), dtype=dtype)
    charges = onp.zeros(shape=(full_size, max_sizes["charges"]), dtype=dtype)
    pairs_14 = onp.zeros(shape=(full_size, max_sizes["pairs_14"], 2), dtype=onp.int32)-1
    charges_14 = onp.zeros(shape=(full_size, max_sizes["charges_14"]), dtype=dtype)
    sigma_14 = onp.zeros(shape=(full_size, max_sizes["sigma_14"]), dtype=dtype)
    epsilon_14 = onp.zeros(shape=(full_size, max_sizes["epsilon_14"]), dtype=dtype)
    scee_14 = onp.zeros(shape=(full_size, max_sizes["scee_14"]), dtype=dtype)
    scnb_14 = onp.zeros(shape=(full_size, max_sizes["scnb_14"]), dtype=dtype)
    disp_coef = onp.zeros(shape=(full_size,), dtype=dtype)
    gamma = onp.zeros(shape=(full_size, max_sizes["gamma"]+1), dtype=dtype)+1 # for masked diagonal terms
    electronegativity = onp.zeros(shape=(full_size, max_sizes["electronegativity"]+1), dtype=dtype)+1
    hardness = onp.zeros(shape=(full_size, max_sizes["hardness"]+1), dtype=dtype)+1
    species = onp.zeros(shape=(full_size, max_sizes["species"]), dtype=onp.int32)-1
    name_to_index = onp.zeros(shape=(full_size,), dtype=onp.int32) # TODO placeholder
    #name_to_index = [None] * full_size
    solute_cut = onp.zeros(shape=(full_size,), dtype=onp.int32)

    for i in range(full_size):
        #TODO should probably change any occurences with item to scalar values in the helper
        # this is a weird situation where there are benefits to having arrays and scalars
        # but 0 dim arrays and scalars aren't really substitutable
        f = force_fields[i]

        name[i] = f.name
        atom_types[i] = f.atom_types
        total_charge[i] = f.total_charge
        atomic_number[i,:len(f.atomic_number)] = f.atomic_number
        #params_to_indices[i] = f.params_to_indices
        bond_restraints[i] = f.bond_restraints
        angle_restraints[i] = f.angle_restraints
        torsion_restraints[i] = f.torsion_restraints
        atom_count[i] = f.atom_count
        positions[i,:f.atom_count,:] = f.positions[:f.atom_count]
        box_vectors[i] = f.box_vectors
        masses[i,:f.atom_count] = f.masses[:f.atom_count]
        cutoff[i] = f.cutoff
        nbr_list[i] = f.nbr_list
        grid_points[i] = f.grid_points
        ewald_alpha[i] = f.ewald_alpha
        ewald_error[i] = f.ewald_error
        dr_threshold[i] = f.dr_threshold
        exclusions[i,:len(f.exclusions),:] = f.exclusions[:len(f.exclusions)]
        bond_idx[i,:len(f.bond_idx),:] = f.bond_idx[:len(f.bond_idx)]
        bond_k[i,:len(f.bond_k)] = f.bond_k[:len(f.bond_k)]
        bond_len[i,:len(f.bond_len)] = f.bond_len[:len(f.bond_len)]
        angle_idx[i,:len(f.angle_idx),:] = f.angle_idx[:len(f.angle_idx)]
        angle_k[i,:len(f.angle_k)] = f.angle_k[:len(f.angle_k)]
        angle_equil[i,:len(f.angle_equil)] = f.angle_equil[:len(f.angle_equil)]
        torsion_idx[i,:len(f.torsion_idx),:] = f.torsion_idx[:len(f.torsion_idx)]
        torsion_k[i,:len(f.torsion_k)] = f.torsion_k[:len(f.torsion_k)]
        torsion_phase[i,:len(f.torsion_phase)] = f.torsion_phase[:len(f.torsion_phase)]
        torsion_period[i,:len(f.torsion_period)] = f.torsion_period[:len(f.torsion_period)]
        pairs[i,:len(f.pairs),:] = f.pairs[:len(f.pairs)]
        sigma[i,:len(f.sigma)] = f.sigma[:len(f.sigma)]
        epsilon[i,:len(f.epsilon)] = f.epsilon[:len(f.epsilon)]
        charges[i,:len(f.charges)] = f.charges[:len(f.charges)]
        pairs_14[i,:len(f.pairs_14),:] = f.pairs_14[:len(f.pairs_14)]
        charges_14[i,:len(f.charges_14)] = f.charges_14[:len(f.charges_14)]
        sigma_14[i,:len(f.sigma_14)] = f.sigma_14[:len(f.sigma_14)]
        epsilon_14[i,:len(f.epsilon_14)] = f.epsilon_14[:len(f.epsilon_14)]
        scee_14[i,:len(f.scee_14)] = f.scee_14[:len(f.scee_14)]
        scnb_14[i,:len(f.scnb_14)] = f.scnb_14[:len(f.scnb_14)]
        disp_coef[i] = f.disp_coef
        gamma[i,:len(f.gamma)] = f.gamma[:len(f.gamma)]
        electronegativity[i,:len(f.electronegativity)] = f.electronegativity[:len(f.electronegativity)]
        hardness[i,:len(f.hardness)] = f.hardness[:len(f.hardness)]
        species[i,:f.atom_count] = f.species[:f.atom_count]
        #name_to_index[i] = f.name_to_index
        solute_cut[i] = f.solute_cut

    new_ff = AmberForceField(
                            name=name,
                            atom_types=atom_types,
                            total_charge=total_charge,
                            atomic_number=atomic_number,
                            params_to_indices=params_to_indices,
                            bond_restraints=bond_restraints,
                            angle_restraints=angle_restraints,
                            torsion_restraints=torsion_restraints,
                            atom_count=atom_count,
                            positions=positions,
                            box_vectors=box_vectors,
                            masses=masses,
                            cutoff=cutoff,
                            nbr_list=nbr_list,
                            grid_points=grid_points,
                            ewald_alpha=ewald_alpha,
                            ewald_error=ewald_error,
                            dr_threshold=dr_threshold,
                            exclusions=exclusions,
                            bond_idx=bond_idx,
                            bond_k=bond_k,
                            bond_len=bond_len,
                            angle_idx=angle_idx,
                            angle_k=angle_k,
                            angle_equil=angle_equil,
                            torsion_idx=torsion_idx,
                            torsion_k=torsion_k,
                            torsion_phase=torsion_phase,
                            torsion_period=torsion_period,
                            pairs=pairs,
                            sigma=sigma,
                            epsilon=epsilon,
                            charges=charges,
                            pairs_14=pairs_14,
                            charges_14=charges_14,
                            sigma_14=sigma_14,
                            epsilon_14=epsilon_14,
                            scee_14=scee_14,
                            scnb_14=scnb_14,
                            disp_coef=disp_coef,
                            gamma=gamma,
                            electronegativity=electronegativity,
                            hardness=hardness,
                            species=species,
                            name_to_index=name_to_index,
                            solute_cut=solute_cut)

    return new_ff

def parse_and_save_force_field_amber(out_name, out_params):
    # args.init_FF, new_name, new_force_field
    # TODO implement more rigorous parser, difficult due to different use cases
    # and the general nature of amber parameter templates
    # a more structured output that doesn't modify force fields might also work
    out_params = onp.array(out_params)
    with open(out_name, 'w') as param:
        for p in out_params:
            param.write(f"{p}\n")