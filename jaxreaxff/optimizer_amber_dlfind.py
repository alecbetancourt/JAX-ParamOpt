import sys, json
from scipy.optimize import minimize
import numpy as np
import jax
#jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax_md
import matplotlib.pyplot as plt
import jax_md.amber.amber_energy as amber
#import parmed as pmd
import openmm as omm
import openmm.app as app
jax.config.update("jax_enable_x64", True)
import argparse
from jaxreaxff.smartformatter import SmartFormatter

from parmedmod import UpdateParmTopCLI
import os, re

# make global array for loss
losses = []
iteration = 0

best_loss = jnp.iinfo(jnp.int64).max
best_params = None
best_iteration = -1

# Reads json data
def ReadJsonData(json_path):

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    return json_data

# dumps json data into an existing file
def SaveJsonData(field_dict, json_path):

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for key in field_dict:
        json_data[key]=field_dict[key]

    with open(json_path, 'w') as outfile:
        json.dump(json_data, outfile)

def extractCoordinates(flist):
    coordinates = []

    for file in flist:
        with open(file, 'r') as f:
            lines=f.readlines()

        crds = []
        for line in lines[2:]:
            crds.append([jnp.float32(i) for i in line.split()[1:]])
        #print(crds)
        crds = jnp.array(crds)
        #A -> NM
        coordinates.append(crds/10)

    return coordinates

def constrained_minimization_vec(crds, prmtop, boxVectors, min_steps, torsions, min_interval):
    radian_to_degree = 180.0/jnp.pi
    degree_to_radian = 1.0/radian_to_degree
    bondprm = amber.bond_init(prmtop._prmtop)
    angleprm = amber.angle_init(prmtop._prmtop)
    torsionprm = amber.torsion_init(prmtop._prmtop)
    ljprm = amber.lj_init(prmtop._prmtop)
    coulprm = amber.coul_init(prmtop._prmtop)
    prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)
    # restraint format:
    # p1,p2,p3,p4,resangle(radians),frc1,frc2
    # default frc1/frc2 values - 1000.0/0.25
    # frc1 = 1000
    # frc2 = .25
    # t = [6,7,9,11]

    def energy_fn(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm) \
                + amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)
                #+ 0)/4.184

    def energy_fn_no_restraint(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm) \
                #+ amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)
                + 0)/4.184)

    masses = jnp.array([jnp.float32(val) for val in prmtop._prmtop._raw_data['MASS']])
    displacement_fn, shift_fn = jax_md.space.periodic_general(boxVectors, fractional_coordinates=False)
    key = jax.random.PRNGKey(0)
    energy_fn = jax.jit(energy_fn)
    init_fn, apply_fn = jax_md.minimize.fire_descent(energy_fn, shift_fn, 1e-3, 1e-3)
    #state = init_fn(positions, mass=masses)

    def body_fn(i, stateList):
        state, restraint, prms = stateList
        state = apply_fn(state, prms=prms, restraint=restraint)
        #return (state, nbpairs)
        return state, restraint, prms

    inner = 100
    outer = int(min_steps/inner)

    initial_torsions = []
    actual_torsions = []
    pre_energies = []
    pre_rest_energies = []
    energies = []
    rest_energies = []
    pairs = []
    mdtimes = []
    post_positions = []

    global iteration
    iteration = iteration + 1

    if iteration % min_interval == 0:
        print("Minimization Run")
        # vmap instead of naive loop
        target_angle = jnp.array([i for i in range(36)])
        crds = jnp.array(crds)
        batch_inner = jax.vmap(min_inner, in_axes=(0, 0, None, None, None, None, None, None, None, None), out_axes=(0,0,0))

        energies, post_positions, actual_torsions = batch_inner(crds, target_angle, energy_fn_no_restraint, energy_fn, init_fn, body_fn, masses, boxVectors, prms, torsions)

        deviation = []
        for i, j in enumerate(actual_torsions):
            current_angle = j * radian_to_degree
            current_angle = jnp.where(current_angle < 0.0, current_angle + 360.0, current_angle)
            deviation.append(jnp.absolute(i*10 - current_angle))

        print("Average angular deviation from restrained angle:", jnp.mean(jnp.array(deviation)))
        print("Individual deviations from restrained angle:", deviation)
    else:
        post_positions = crds
        energies = jnp.array([energy_fn_no_restraint(p, prms=prms) for p in crds])

    return energies, post_positions

def min_inner(crds, target, energy_fn_no_restraint, energy_fn, init_fn, body_fn, masses, boxVectors, prms, torsions):
    radian_to_degree = 180.0/jnp.pi
    degree_to_radian = 1.0/radian_to_degree
    frc1 = 1000
    frc2 = 0.1
    t = torsions[0]

    target_angle = target * 10 * degree_to_radian
    curr_rest = [t[0],t[1],t[2],t[3], target_angle, frc1, frc2]

    current_crds = crds

    pre_energies = energy_fn_no_restraint(current_crds, prms=prms)

    state = init_fn(current_crds, mass=masses, restraint=curr_rest, prms=prms)

    state = jax_md.minimize.FireDescentState(jnp.float64(state.position),jnp.float64(state.momentum),\
                                                jnp.float64(state.force), state.mass, state.dt, state.alpha,\
                                                state.n_pos)

    p1 = state.position[t[0]]
    p2 = state.position[t[1]]
    p3 = state.position[t[2]]
    p4 = state.position[t[3]]
    initial_torsions = amber.torsion_single(p1,p2,p3,p4, boxVectors)

    iter = 2000
    inner = 100
    outer = int(iter/inner)

    for i in range(outer):
        state, curr_rest, prms = jax.lax.fori_loop(0, inner, body_fn, (state, curr_rest, prms))

    p1 = state.position[t[0]]
    p2 = state.position[t[1]]
    p3 = state.position[t[2]]
    p4 = state.position[t[3]]
    actual_torsions = amber.torsion_single(p1,p2,p3,p4, boxVectors)

    post_positions = state.position
    energies = energy_fn_no_restraint(state.position, prms=prms)

    return energies, post_positions, actual_torsions

def gradObj(scipy_params, *args):
    crds, boxVectors, ref_ene, post_positions, prms_pre, torsions = args

    prms = prms_pre

    i = 0
    for idx in torsions[:, 4]:
        prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][idx] = scipy_params[i]
        prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][idx] = scipy_params[i+1]
        prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][idx] = scipy_params[i+2]
        i = i + 3

    bondprm = amber.bond_init(prms._prmtop)
    angleprm = amber.angle_init(prms._prmtop)
    torsionprm = amber.torsion_init(prms._prmtop)
    ljprm = amber.lj_init(prms._prmtop)
    coulprm = amber.coul_init(prms._prmtop)
    prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)
    def energy_fn(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm))/4.184)
                #+ amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)

    ene_list = [energy_fn(p, prms=prms) for p in post_positions]

    min_ene = min(ene_list)

    relative_ene_list = [(x - min_ene) for x in ene_list]

    np_relative_ene_list=jnp.array(relative_ene_list)
    np_ref_ene=jnp.array(ref_ene)

    difference=np_ref_ene-np_relative_ene_list
    nrg_diff_sqrd = jnp.sum(difference ** 2)

    return nrg_diff_sqrd, relative_ene_list

# updates amber prmtop file, runs constrained optimizations, computes difference between ref and computed energy profiles and RMSD.
def ObjectiveFunction(scipy_params, *args):
    global iteration
    iteration = iteration + 1
    print("Iteration:", iteration)

    crds, boxVectors, ref_ene, params_dict, optvars_dict, prms, torsions, min_steps, outdir, prmtop_dir, min_interval, crd_flist, geo_dir, amber_dir = args

    print("Updated Parameters:", scipy_params)

    # set new parameters
    i=0
    for key, value in params_dict.items():
        if(optvars_dict['height']):
            value['height']=scipy_params[i]
            i+=1

        if(optvars_dict['phase']):
            value['phase']=scipy_params[i]
            i+=1

        if(optvars_dict['periodicity']):
            value['periodicity']=scipy_params[i]
            i+=1

        if(optvars_dict['scee']):
            value['scee']=scipy_params[i]
            i+=1

        if(optvars_dict['scnb']):
            value['scnb']=scipy_params[i]
            i+=1

    # UpdateParmTopCLI(prmtop_dir, params_dict)

    # update parmtop file with new parameters
    #dh_dir='./confs_999-999/dh_7-9-10-11/'
    UpdateParmTopCLI(prmtop_dir, params_dict)

    i = 0
    for idx in torsions[:, 4]:
        prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][idx] = scipy_params[i]
        prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][idx] = scipy_params[i+1]
        prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][idx] = scipy_params[i+2]
        i = i + 3

    # ene_list, post_positions = constrained_minimization_vec(crds, prms, boxVectors, min_steps, torsions, min_interval)

    # TODO make sure this below works
    # if interval mod current interval = 0:
    # make sure to add this option again
    # run torsional scan calculations using geometric and sander
    #print("Iter", iteration)
    #print("cut", iteration-1 % min_interval)
    #print("cutp", iteration-1 % min_interval == 0)

    if (iteration-1) % min_interval == 0:
        ierr = os.system('cd %s && rm -rf *.tmp *.log *.out *_optim* *.restrt *.path* *.rst7 *_post.xyz' % (amber_dir))
        run_task_command="""bash run_task_scipyopt_0.sh > run_task_scipyopt_0.log 2>&1 &
        pid=$!
        echo $pid > run_task.pid
        wait $pid
        """
        ierr = os.system(run_task_command)
        if(ierr != 0):
            print('Error: Please check the run.log file.')
            return

    # extract data
    ene_list=list()
    output_flist=[geo_dir + '_%03d' % (i) + '.out' for i in range(36)]
    #TODO: change this
    for fname in output_flist:
        with open(fname, 'r') as f:
            lines=f.readlines()

        ene=[float(re.findall(r'(\-*\d+\.\d+)',l)[0]) for l in lines if re.match(r'.*Final converged energy:.*',l)]

        if(len(ene) == 0):
            print('Error: Failed to extract energy from %s.' % (fname))
            return

        ene_list.append(ene[-1])

    crd_flist=[geo_dir + '_%03d' % (i) + '_post.xyz' for i in range(36)]
    post_positions = extractCoordinates(crd_flist)
    #TODO make sure you compare amber energies and the ones generated by this

    loss_and_grad_fn = jax.value_and_grad(gradObj, has_aux=True)
    loss_and_grad = loss_and_grad_fn(scipy_params, crds, boxVectors, ref_ene, post_positions, prms, torsions)

    #global iteration
    #iteration = iteration + 1
    #print("Extracted energy list:", ene_list)

    loss_ene, grad = loss_and_grad
    loss, jax_ene_list = loss_ene
    grad = grad.astype('float64')
    print("Loss", loss)
    losses.append(loss)
    print("Loss Grad", grad)

    #TODO test SSE against RMSD
    #also check internal jax energies against this
    #ie print(ene_list, ene_list_jax) and diff
    #ene_list_jax = [energy_fn(p, prms=prms) for p in post_positions]
    min_ene = min(ene_list)
    relative_ene_list = [(x - min_ene) * 627.5 for x in ene_list]

    global gaff_ene_list
    if iteration == 1:
        gaff_ene_list = relative_ene_list

    plt.plot(range(0,360,10), ref_ene, marker='o', label="Reference")
    #plt.plot(range(0,360,10), relative_ene_list, marker='o', label="Sander Energies Post Optimization")
    plt.plot(range(0,360,10), gaff_ene_list, marker='o', label="Initial Guess")
    plt.plot(range(0,360,10), jax_ene_list, marker='o', label="Current Guess")
    plt.title("JAX-AMBER + DLFind Fitting Iteration %s" % iteration)
    plt.xlabel("Dihedral (Degree)")
    plt.ylabel("Potential Energy (kcal/mol)")
    plt.legend()
    plt.savefig(outdir + "/iteration_%s.png" % iteration)
    plt.close()

    # Update best values if applicable
    global best_loss
    global best_params
    global best_iteration
    if loss < best_loss:
        best_loss = loss
        best_params = scipy_params
        best_iteration = iteration

    # scipy requires jac gradient as list
    # print("loss", loss)
    # print("grad", grad)
    #sys.exit()
    return loss, list(grad)

def ff_opt(prmtop_dir, params_dir, geo_dir, amber_dir, min_steps, opt_loops, ref_ene, outdir, min_interval):
    initial_guess='initial_guess'
    algorithm='L-BFGS-B'
    # maxiter=1000
    step_size=0.100000

    crd_flist=[geo_dir + '_%03d' % (i) + '.xyz' for i in range(36)]

    #list of 36 (35,3) numpy arrays from 0-350 deg
    coordinates = extractCoordinates(crd_flist)

    params_dict=ReadJsonData(params_dir)[initial_guess]
    optvars_dict=ReadJsonData(params_dir)['optvars']
    bounds_dict=ReadJsonData(params_dir)['bounds']

    guess=list()
    bounds=list()

    for key, value in params_dict.items():
        if(optvars_dict['height']):
            guess.append(value['height'])
            bounds.append(bounds_dict['height'])

        if(optvars_dict['phase']):
            guess.append(value['phase'])
            bounds.append(bounds_dict['phase'])

        if(optvars_dict['periodicity']):
            guess.append(value['periodicity'])
            bounds.append(bounds_dict['periodicity'])

        if(optvars_dict['scee']):
            guess.append(value['scee'])
            bounds.append(bounds_dict['scee'])

        if(optvars_dict['scnb']):
            guess.append(value['scnb'])
            bounds.append(bounds_dict['scnb'])

    # Make initial FF modifications using parmed
    rng = np.random.default_rng()
    for k in params_dict:
        params_dict[k]['height'] += rng.random() # * 1e-5 too small of a value and parmed will truncate it, adjust this if you'd like
    UpdateParmTopCLI(prmtop_dir, params_dict)

    prmtopomm = app.AmberPrmtopFile(prmtop_dir)

    # Grab all indices of torsions from the params file
    torsions = [list(map(int, torsion.split("-"))) for torsion in params_dict.keys()]
    ref_ene = jnp.array(ReadJsonData(ref_ene)['ref_ene'])
    print("Torsion Indices from parameter file:", torsions)

    # Use regular numpy to prevent tracing to make this easier
    torsionidx = prmtopomm._prmtop._raw_data["DIHEDRALS_INC_HYDROGEN"] + prmtopomm._prmtop._raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
    torsionidx = np.array([int(index) for index in torsionidx]).reshape((-1,5))
    torsionidx[:, :4] = torsionidx[:, :4]//3
    torsionidx[:, 4] = torsionidx[:, 4]-1
    print("All Torsion Indices:", torsionidx)

    # Find the actual parameter index in the prmtop file using the atom numbers for the torsion
    torsion_indices = []
    torsion_idx_list = torsionidx.tolist()
    for torsion in torsions:
        for torsion_idx in torsion_idx_list:
            if torsion == torsion_idx[:4]:
                torsion_indices.append(torsion_idx)

    torsion_indices = jnp.array(torsion_indices)

    print("Selected Torsion & Parameter Indices:")
    print(torsion_indices)

    print("Torsion to be constrained:", torsions[0][:4])

    # sys.exit()

    #system = prmtopomm.createSystem(nonbondedMethod=app.NoCutoff, removeCMMotion=False, constraints=None)
    #boxVectors = jnp.array([v._value for v in system.getDefaultPeriodicBoxVectors()])
    #boxVectors = boxVectors.sum(axis=0)
    boxVectors = jnp.array([100.0, 100.0, 100.0])

    minimization_result=minimize(ObjectiveFunction, guess, jac=True, \
           args=(coordinates, boxVectors, ref_ene, params_dict, optvars_dict, prmtopomm, torsion_indices, 
                 min_steps, outdir, prmtop_dir, min_interval, crd_flist, geo_dir, amber_dir), \
           bounds=bounds, method=algorithm, options={'maxiter':opt_loops, 'eps': step_size})

    print("Losses:", losses)

    print("Best Loss:", best_loss)

    print("Best Iteration:", best_iteration)

    print("Best Params:", best_params)

    print("Final Params: ", minimization_result.x)

    print("Termination Message: ", minimization_result.message)

    x = minimization_result.x

    # Set final parameters in dictionary and save
    i=0
    for key, value in params_dict.items():
        if(optvars_dict['height']):
            value['height']=x[i]
            i+=1

        if(optvars_dict['phase']):
            value['phase']=x[i]
            i+=1

        if(optvars_dict['periodicity']):
            value['periodicity']=x[i]
            i+=1

        if(optvars_dict['scee']):
            value['scee']=x[i]
            i+=1

        if(optvars_dict['scnb']):
            value['scnb']=x[i]
            i+=1

    # SaveJsonData(params_dict, outdir + '/final_params.json')

    return

def main():
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='AMBER Torsion Optimizer',
                                   formatter_class=SmartFormatter)

    parser.add_argument('--prmtop', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/prmtop",
      help='Location of PRMTOP file')
    parser.add_argument('--params', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/params.json",
      help='Parameters file')
    parser.add_argument('--geo', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/confs_999-999/dh_6-7-9-11/dh_6-7-9-11",
      help='Directory with geometry files')
    parser.add_argument('--amber_dir', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/confs_999-999/dh_6-7-9-11",
      help='Directory with AMBER files')
    parser.add_argument('--reference', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/ref_ene.json",
      help='Directory to output results')
    parser.add_argument('--out', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/jaxout",
      help='Directory to output results')
    parser.add_argument('--minsteps', metavar='steps',
      type=int,
      default=2000,
      help='Maximum number of energy minimization steps')
    parser.add_argument('--maxiter', metavar='iterations',
      type=int,
      default=1000,
      help='Maximum number of optimization iterations')
    parser.add_argument('--mininterval', metavar='iterations',
      type=int,
      default=5,
      help='Number of parameter optimization iterations between geometry optimization')

    args = parser.parse_args()

    ff_opt(args.prmtop, args.params, args.geo, args.amber_dir, args.minsteps, args.maxiter, args.reference, args.out, args.mininterval)

if __name__ == "__main__":
    main()

# needs a file with the reference energies as a dictionary, very simple format
# the torsions are read from the params file and then matched with the parameter indices from the prmtop
# it's assumed that the torsion being constrained is the first torsion in this list but i could change this if some other behavior is desired

# have to figure out how to do parmed mods for torsion prms, if 2 torsions have the same params, they end up mapping to the same index, even when parmed updates them
# this isn't good because we want different indices for every torsion, not sure if there's an easy way to force seperate parameter indices to be generated
# the code as is doesn't touch parmed except for the final parameter update so it's assumed that the seperate torsion parameter indices exist before running the optimizer


# if any structures don't display good results, we can look into changing minimization interval, optimizer tolerance, and a few other
# things. there was also the discussion about offloading the constrained minimization to a package with better tools for it and just
# doing the final gradient evaluation in jax at a potential speed hit. constraint parameters can also be tuned with current reax style approach

# could look into doing meta optimization of these penalty term parameters assuming there's good energy or other physical references
# from a better approach to avoid overfitting the restraint by treating loss as angular deviation alone
