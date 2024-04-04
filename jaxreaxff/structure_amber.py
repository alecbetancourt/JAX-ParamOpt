from jax_md import dataclasses, util
import numpy as onp
Array = util.Array
import jax_md.amber.amber_energy as amber
#from jaxreaxff.generate_prmtop import build_prm_list
import openmm as omm

#bond angle torsion lj coul init
#(k, l, b1_idx, b2_idx, param_index)
#(k, eqangle, a1_idx, a2_idx, a3_idx, param_index)
#(k, phase, periodicity, t1_idx, t2_idx, t3_idx, t4_idx, param_index)
#(pairs, pairs14, atom_type, sigma, epsilon, scnb)
#(charges, pairs, pairs14, scee)

@dataclasses.dataclass
class AmberForceField(object):
    # name: Array
    # atom_count: Array
    # atom_types: Array
    # atomic_nums: Array
    # positions: Array
    # orth_matrix: Array

    # total_charge: Array
    # energy_minimize: Array
    # energy_minim_steps: Array
    # periodic_image_shifts: Array

    # bond_restraints: BondRestraint
    # angle_restraints: AngleRestraint
    # torsion_restraints: TorsionRestraint

    # target_e: Array
    # target_f: Array
    # target_ch: Array
    
    # TODO: Create numeric mapping scheme to populate full forcefield in JAX friendly way

    # Bond Parameters
    b_k: Array
    b_l: Array
    b_1_idx: Array
    b_2_idx: Array
    b_prm_idx: Array

    # Angle Parameters
    a_k: Array
    a_eq_ang: Array
    a_1_idx: Array
    a_2_idx: Array
    a_3_idx: Array
    a_prm_idx: Array

    # Torsion Parameters
    t_k: Array
    t_phase: Array
    t_period: Array
    t_1_idx: Array
    t_2_idx: Array
    t_3_idx: Array
    t_4_idx: Array
    t_prm_idx: Array

    # Common Nonbonded Parameters
    pairs: Array
    pairs14: Array

    # Lennard-Jones Parameters
    lj_type: Array
    sigma: Array
    epsilon: Array
    scnb: Array

    # Coulomb Parameters
    charges: Array
    scee: Array



# use onp for all this to avoid tracing
# likely need to eventually record lengths instead of applying a mask to keep track of unused indices
def align_forcefield(structures, max_sizes, dtype=onp.float32):
    full_size = len(structures)
    #num_atoms = max_sizes['num_atoms']

    #b_k = onp.zeros(shape=(full_size,max_sizes['b_k']), dtype=dtype)
    #b_l = onp.zeros(shape=(full_size,max_sizes['b_l']), dtype=dtype)
    b_k = onp.zeros(shape=(full_size,max_sizes['b_k']+1), dtype=dtype)
    b_l = onp.zeros(shape=(full_size,max_sizes['b_l']+1), dtype=dtype)
    #b_1_idx = onp.zeros(shape=(full_size,max_sizes['b_1_idx']+1), dtype=onp.int32)
    #b_2_idx = onp.zeros(shape=(full_size,max_sizes['b_2_idx']+1), dtype=onp.int32)
    b_1_idx = onp.full(shape=(full_size,max_sizes['b_1_idx']), fill_value=-1, dtype=onp.int32)
    b_2_idx = onp.full(shape=(full_size,max_sizes['b_2_idx']), fill_value=-1, dtype=onp.int32)
    b_prm_idx = onp.full(shape=(full_size,max_sizes['b_prm_idx']), fill_value=-1, dtype=onp.int32)
    #b_prm_idx = onp.zeros(shape=(full_size,max_sizes['b_prm_idx']), dtype=onp.int32)

    a_k = onp.zeros(shape=(full_size,max_sizes['a_k']+1), dtype=dtype)
    a_eq_ang = onp.zeros(shape=(full_size,max_sizes['a_eq_ang']+1), dtype=dtype)
    # a_1_idx = onp.zeros(shape=(full_size,max_sizes['a_1_idx']), dtype=onp.int32)
    # a_2_idx = onp.zeros(shape=(full_size,max_sizes['a_2_idx']), dtype=onp.int32)
    # a_3_idx = onp.zeros(shape=(full_size,max_sizes['a_3_idx']), dtype=onp.int32)
    # a_prm_idx = onp.zeros(shape=(full_size,max_sizes['a_prm_idx']), dtype=onp.int32)
    a_1_idx = onp.full(shape=(full_size,max_sizes['a_1_idx']), fill_value=-1, dtype=onp.int32)
    a_2_idx = onp.full(shape=(full_size,max_sizes['a_2_idx']), fill_value=-1, dtype=onp.int32)
    a_3_idx = onp.full(shape=(full_size,max_sizes['a_3_idx']), fill_value=-1, dtype=onp.int32)
    a_prm_idx = onp.full(shape=(full_size,max_sizes['a_prm_idx']), fill_value=-1, dtype=onp.int32)

    t_k = onp.zeros(shape=(full_size,max_sizes['t_k']+1), dtype=dtype)
    t_phase = onp.zeros(shape=(full_size,max_sizes['t_phase']+1), dtype=dtype)
    t_period = onp.zeros(shape=(full_size,max_sizes['t_period']+1), dtype=dtype)
    # t_1_idx = onp.zeros(shape=(full_size,max_sizes['t_1_idx']), dtype=onp.int32)
    # t_2_idx = onp.zeros(shape=(full_size,max_sizes['t_2_idx']), dtype=onp.int32)
    # t_3_idx = onp.zeros(shape=(full_size,max_sizes['t_3_idx']), dtype=onp.int32)
    # t_4_idx = onp.zeros(shape=(full_size,max_sizes['t_4_idx']), dtype=onp.int32)
    # t_prm_idx = onp.zeros(shape=(full_size,max_sizes['t_prm_idx']), dtype=onp.int32)
    t_1_idx = onp.full(shape=(full_size,max_sizes['t_1_idx']), fill_value=-1, dtype=onp.int32)
    t_2_idx = onp.full(shape=(full_size,max_sizes['t_2_idx']), fill_value=-1, dtype=onp.int32)
    t_3_idx = onp.full(shape=(full_size,max_sizes['t_3_idx']), fill_value=-1,dtype=onp.int32)
    t_4_idx = onp.full(shape=(full_size,max_sizes['t_4_idx']), fill_value=-1, dtype=onp.int32)
    t_prm_idx = onp.full(shape=(full_size,max_sizes['t_prm_idx']), fill_value=-1, dtype=onp.int32)

    # pairs = onp.zeros(shape=(full_size,max_sizes['pairs'],2), dtype=onp.int32)
    # pairs14 = onp.zeros(shape=(full_size,max_sizes['pairs14'],3), dtype=onp.int32)
    pairs = onp.full(shape=(full_size,max_sizes['pairs'],2), fill_value=-1, dtype=onp.int32)
    pairs14 = onp.full(shape=(full_size,max_sizes['pairs14'],3), fill_value=-1, dtype=onp.int32)

    # lj_type = onp.zeros(shape=(full_size, max_sizes['lj_type']), dtype=onp.int32)
    lj_type = onp.full(shape=(full_size, max_sizes['lj_type']), fill_value=-1, dtype=onp.int32)
    sigma = onp.zeros(shape=(full_size,max_sizes['sigma']+1), dtype=dtype)
    epsilon = onp.zeros(shape=(full_size,max_sizes['epsilon']+1), dtype=dtype)
    scnb = onp.zeros(shape=(full_size,max_sizes['scnb']+1), dtype=dtype)+1 # for masking
    
    charges = onp.zeros(shape=(full_size,max_sizes['charges']+1), dtype=dtype)
    scee = onp.zeros(shape=(full_size,max_sizes['scee']+1), dtype=dtype)+1 # for masking

    #store box vectors?

    for i in range(full_size):
        s = structures[i]

        b_k[i,:len(s['b_k'])] = s['b_k']
        b_l[i,:len(s['b_l'])] = s['b_l']
        b_1_idx[i,:len(s['b_1_idx'])] = s['b_1_idx']
        b_2_idx[i,:len(s['b_2_idx'])] = s['b_2_idx']
        b_prm_idx[i,:len(s['b_prm_idx'])] = s['b_prm_idx']

        a_k[i,:len(s['a_k'])] = s['a_k']
        a_eq_ang[i,:len(s['a_eq_ang'])] = s['a_eq_ang']
        a_1_idx[i,:len(s['a_1_idx'])] = s['a_1_idx']
        a_2_idx[i,:len(s['a_2_idx'])] = s['a_2_idx']
        a_3_idx[i,:len(s['a_3_idx'])] = s['a_3_idx']
        a_prm_idx[i,:len(s['a_prm_idx'])] = s['a_prm_idx']

        t_k[i,:len(s['t_k'])] = s['t_k']
        t_phase[i,:len(s['t_phase'])] = s['t_phase']
        t_period[i,:len(s['t_period'])] = s['t_period']
        t_1_idx[i,:len(s['t_1_idx'])] = s['t_1_idx']
        t_2_idx[i,:len(s['t_2_idx'])] = s['t_2_idx']
        t_3_idx[i,:len(s['t_3_idx'])] = s['t_3_idx']
        t_4_idx[i,:len(s['t_4_idx'])] = s['t_4_idx']
        t_prm_idx[i,:len(s['t_prm_idx'])] = s['t_prm_idx']

        pairs[i,:len(s['pairs']),:] = s['pairs']
        pairs14[i,:len(s['pairs14']),:] = s['pairs14']

        lj_type[i,:len(s['lj_type'])] = s['lj_type']
        sigma[i,:len(s['sigma'])] = s['sigma']
        epsilon[i,:len(s['epsilon'])] = s['epsilon']
        scnb[i,:len(s['scnb'])] = s['scnb']

        charges[i,:len(s['charges'])] = s['charges']
        scee[i,:len(s['scee'])] = s['scee']

    new_system = AmberForceField(b_k=b_k,
                                b_l=b_l,
                                b_1_idx=b_1_idx,
                                b_2_idx=b_2_idx,
                                b_prm_idx=b_prm_idx,
                                a_k=a_k,
                                a_eq_ang=a_eq_ang,
                                a_1_idx=a_1_idx,
                                a_2_idx=a_2_idx,
                                a_3_idx=a_3_idx,
                                a_prm_idx=a_prm_idx,
                                t_k=t_k,
                                t_phase=t_phase,
                                t_period=t_period,
                                t_1_idx=t_1_idx,
                                t_2_idx=t_2_idx,
                                t_3_idx=t_3_idx,
                                t_4_idx=t_4_idx,
                                t_prm_idx=t_prm_idx,
                                pairs=pairs,
                                pairs14=pairs14,
                                lj_type=lj_type,
                                sigma=sigma,
                                epsilon=epsilon,
                                scnb=scnb,
                                charges=charges,
                                scee=scee)

    return new_system

# pass prm list, load each ff and record prm lengths for each field
# load ffs into list of dictionaries where each dict represents one prmtop
# return ff list and a dict of all max sizes for alignment
# eventually offload the dictionary generation portion to the prmtop loader
# and replace this with a slimmed down function
def load_ff(prm_list):
    #max k
    #max b len = 0
    #prmdictlist = []
    max_sizes = {}
    #size_keys = 
    #max_sizes = dict.fromkeys(, 0)

    max_sizes["b_k"] = 0
    max_sizes["b_l"] = 0
    max_sizes["b_1_idx"] = 0
    max_sizes["b_2_idx"] = 0
    max_sizes["b_prm_idx"] = 0

    max_sizes["a_k"] = 0
    max_sizes["a_eq_ang"] = 0
    max_sizes["a_1_idx"] = 0
    max_sizes["a_2_idx"] = 0
    max_sizes["a_3_idx"] = 0
    max_sizes["a_prm_idx"] = 0

    max_sizes["t_k"] = 0
    max_sizes["t_phase"] = 0
    max_sizes["t_period"] = 0
    max_sizes["t_1_idx"] = 0
    max_sizes["t_2_idx"] = 0
    max_sizes["t_3_idx"] = 0
    max_sizes["t_4_idx"] = 0
    max_sizes["t_prm_idx"] = 0

    max_sizes["pairs"] = 0
    max_sizes["pairs14"] = 0

    max_sizes["lj_type"] = 0
    max_sizes["sigma"] = 0
    max_sizes["epsilon"] = 0
    max_sizes["scnb"] = 0

    max_sizes["charges"] = 0
    max_sizes["scee"] = 0

    #for each struct
    #prmdict = {}
    #binit, ainit, etc

    prm_dict_list = []
    #print(flist)
    for f in prm_list:
        prm_dict = {}
        prmtop = omm.app.AmberPrmtopFile(f)

        #(k, l, b1_idx, b2_idx, param_index)
        #(k, eqangle, a1_idx, a2_idx, a3_idx, param_index)
        #(k, phase, periodicity, t1_idx, t2_idx, t3_idx, t4_idx, param_index)
        #(pairs, pairs14, atom_type, sigma, epsilon, scnb)
        #(charges, pairs, pairs14, scee)
        b_k, b_l, b_1_idx, b_2_idx, b_prm_idx = amber.bond_init(prmtop._prmtop)
        a_k, a_eq_ang, a_1_idx, a_2_idx, a_3_idx, a_prm_idx = amber.angle_init(prmtop._prmtop)
        t_k, t_phase, t_period, t_1_idx, t_2_idx, t_3_idx, t_4_idx, t_prm_idx = amber.torsion_init(prmtop._prmtop)
        pairs, pairs14, lj_type, sigma, epsilon, scnb = amber.lj_init(prmtop._prmtop)
        charges, pairs, pairs14, scee = amber.coul_init(prmtop._prmtop)

        prm_dict["b_k"] = b_k
        prm_dict["b_l"] = b_l
        prm_dict["b_1_idx"] = b_1_idx
        prm_dict["b_2_idx"] = b_2_idx
        prm_dict["b_prm_idx"] = b_prm_idx

        prm_dict["a_k"] = a_k
        prm_dict["a_eq_ang"] = a_eq_ang
        prm_dict["a_1_idx"] = a_1_idx
        prm_dict["a_2_idx"] = a_2_idx
        prm_dict["a_3_idx"] = a_3_idx
        prm_dict["a_prm_idx"] = a_prm_idx

        prm_dict["t_k"] = t_k
        prm_dict["t_phase"] = t_phase
        prm_dict["t_period"] = t_period
        prm_dict["t_1_idx"] = t_1_idx
        prm_dict["t_2_idx"] = t_2_idx
        prm_dict["t_3_idx"] = t_3_idx
        prm_dict["t_4_idx"] = t_4_idx
        prm_dict["t_prm_idx"] = t_prm_idx

        prm_dict["pairs"] = pairs
        prm_dict["pairs14"] = pairs14

        prm_dict["lj_type"] = lj_type
        prm_dict["sigma"] = sigma
        prm_dict["epsilon"] = epsilon
        prm_dict["scnb"] = scnb

        prm_dict["charges"] = charges
        prm_dict["scee"] = scee

        # TODO: Convert orthogonolization matrix
        prm_dict["boxVectors"] = [999.9,999.9,999.9]

        for key in max_sizes:
            max_sizes[key] = len(prm_dict[key]) if len(prm_dict[key]) > max_sizes[key] else max_sizes[key]

        prm_dict_list.append(prm_dict)

    return prm_dict_list, max_sizes