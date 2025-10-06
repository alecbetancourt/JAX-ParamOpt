#      #sys.exit()
#       ############################################################################################
#       # # alternate approach using thread pool
#       # from concurrent.futures import ProcessPoolExecutor
#       # import multiprocessing

#       # #opt_fn = jax.jit(dlfind_callback) # TODO check jit compilation messages and test device=CPU parameter

#       # # opt_fn = dlfind_min

#       # # opt_fn = jax.tree_util.Partial(opt_fn, ffq_ff=ffq_ff, max_iter=minim_steps)

#       # # opt_fn = lambda args: opt_fn(*args)

#       # batch_size = len(list_sub_cur_pos[i]) # TODO add padding to make all batches evenly sized?

#       # coords_list = [onp.array(list_sub_cur_pos[i][j]) for j in range(batch_size)] # TODO onp for these three?
      
#       # struct_list = [
#       #   move_dataclass(jax.tree_util.tree_map(lambda x: x[j], list_sub_structure[i]), onp)
#       #   for j in range(batch_size)
#       # ]

#       # ff_list = [
#       #   move_dataclass(jax.tree_util.tree_map(lambda x: x[j], force_field[i]), onp)
#       #   for j in range(batch_size)
#       # ]

#       # nbr_list = [None] * batch_size

#       # ffq_ff = move_dataclass(ffq_ff, onp)

#       # minim_steps = onp.int32(minim_steps)

#       # # ffq_list = [ffq_ff] * batch_size
#       # # minim_list = [minim_steps] * batch_size
#       # # iterable_prms = list(zip(coords_list, struct_list, nbr_list, ff_list, ffq_list, minim_list))
      
#       # iterable_prms = list(zip(coords_list, struct_list, nbr_list, ff_list))
#       # print(len(iterable_prms))

#       # # iterable_prms = []
#       # # for j in range(len(list_sub_cur_pos[i])):
#       # #   prms = (list_sub_cur_pos[i][j],
#       # #           tuple(getattr(x, name) for name in list_sub_structure[i][j].data_fields),
#       # #           None,
#       # #           tuple(getattr(x, name) for name in force_field[i][j].data_fields),
#       # #           )
#       # #   iterable_prms.append(prms)

#       # # TODO moving this into the callback itself is probably a better idea
#       # # nvm, callback likely isn't needed either, function isn't jax transformed
#       # # can you treemap over executor.submit to avoid this code above?
#       # with ProcessPoolExecutor(mp_context=multiprocessing.get_context('forkserver'), max_workers=3, initializer=subprocess_init) as executor:
#       #   # results = list(executor.map(opt_fn, list_sub_cur_pos[i],
#       #   #                                     list_sub_structure[i],
#       #   #                                     None, #sub_nbr,
#       #   #                                     force_field[i]))
        
#       #   # results = list(executor.map(opt_fn, iterable_prms))

#       #   # results = list(executor.map(dlfind_min, iterable_prms))

#       #   futures = [executor.submit(dlfind_min,
#       #                               c_single,
#       #                               s_single,
#       #                               n_single,
#       #                               f_single,
#       #                               ffq_ff,
#       #                               minim_steps) for c_single, s_single, n_single, f_single in iterable_prms]

#       #   results = [future.result() for future in futures]

#       # traj_coordinates, traj_energies, final_grad = zip(*results)
#       # traj_coordinates = jnp.stack(traj_coordinates) # TODO onp or jnp? had as onp originally
#       # traj_energies = jnp.stack(traj_energies)
#       # final_grad = jnp.stack(final_grad)

#       # #sys.exit()

#       ###########################################################################################

#             # result_shape_crd = jax.ShapeDtypeStruct(list_sub_cur_pos[i].shape, list_sub_cur_pos[i].dtype)
#       # result_shape_nrg = jax.ShapeDtypeStruct(list_sub_cur_pos[i].shape, list_sub_cur_pos[i].dtype)
#       # result_shape_
#       # result_shape =  (
#       #       jax.ShapeDtypeStruct(list_sub_cur_pos[i].shape, list_sub_cur_pos[i].dtype), # optimized coordinates
#       #       jax.ShapeDtypeStruct((), list_sub_cur_pos[i].dtype),  # final energy scalar
#       #       jax.ShapeDtypeStruct(list_sub_cur_pos[i].shape, list_sub_cur_pos[i].dtype),  # final forces
#       #   )
#       # TODO how to vectorize?
#       #dlf_jit = jax.jit(dlfind_callback) # TODO does this work or result in a speedup? jit of vmap or vmap of jit?
#       #opt_fn = jax.vmap(dlfind_callback, in_axes=(0,0,0,0,None,None))
#       ############################################################################
#       # # working vmap approach but doesn't result in very good performance because of sequential execution
#       # opt_fn = jax.jit(jax.vmap(dlfind_callback, in_axes=(0,0,0,0,None,None)))

#       # #TODO this is likely in kj/mol not kcal
#       # #this probably affects the magnitude of the forces as well as
#       # #
#       # traj_coordinates, traj_energies, final_grad = opt_fn(list_sub_cur_pos[i],
#       #                                                       list_sub_structure[i],
#       #                                                       None, #sub_nbr,
#       #                                                       force_field[i],
#       #                                                       ffq_ff,
#       #                                                       minim_steps)
#       ############################################################################################
#       ### serialization code for separate process optimization

#       # def dlfind_callback(pos, struct, nbrs, ff, ffq_ff, max_iter):
# #   result_shape =  (
# #             jax.ShapeDtypeStruct(pos.shape, pos.dtype), # optimized coordinates
# #             jax.ShapeDtypeStruct((), pos.dtype),  # final energy scalar
# #             jax.ShapeDtypeStruct(pos.shape, pos.dtype),  # final forces
# #   )
# #   # TODO how to vectorize?
# #   #opt_fn = jax.vmap(jax.pure_callback(dlfind_min, result_shape, x, vmap_method='sequential'), in_axes=(0,0,0,0,None,None))
# #   coords, energy, grad = jax.pure_callback(dlfind_min, result_shape, pos, struct, nbrs, ff, ffq_ff, max_iter, vmap_method='sequential')

# #   return coords, energy, grad

#   # with open(filename, "rb") as f:
#   #   data = f.read()
#   # return from_bytes(ForceFieldParams, data)

#   # with open(filename, "rb") as f:
#   #   leaves, treedef = pickle.load(f)
#   # new_pytree = jax.tree_util.tree_unflatten(treedef, leaves)
#   # return new_pytree

#     # state_dict = {
#   #   name: serialization.to_state_dict(getattr(x, name))
#   #   for name in data_fields
#   # }
#   # #return state_dict
#   # data_bytes = to_bytes(dc)
#   # with open(filename, "wb") as f:
#   #   f.write(data_bytes)

#   # leaves, treedef = jax.tree_util.tree_flatten(dc)
#   # with open(filename, "wb") as f:
#   #   pickle.dump((leaves, treedef), f)

# #####################################################################################################################
#   # TODO some important notes about this above option
#   '''
#   GAFF is parameterized the following way - looking at gaff.dat,
#   for all ~70 GAFF types
#   atomic mass values are given according to gaff type - should not be modified

#   bonds are defined as the unique combination of 2 gaff types
#   angles are definfed as the unique combination of 3 gaff types
#   torsions are defined using either a unique combination of the center 2 atom types,
#   or the unique combination of 4 gaff types
#   vdw parameters are specified on a per-gaff atom type basis

#   for ffq, parameters are defined according to gaff atom type

#   in effect, this means that optimizing bonds/angles/torsions in group mode requires mapping to the
#   underlying gaff pairs, which is difficult and may be expensive
#   mapping indices or fully regenerating the prmtop with leap are two of the only options that i can think of for this

#   vdw and ffq should be quite a bit easier as there just needs to be a common type->index enum in the helper file
#   in single mode, things are easier in general as there just needs to be an additional flag for which file a parameter maps to
#   and then the bond/angle/torsion masks can be stored as static dictionaries

#   so for the first implementation:
#   single mode:
#   bond/angle/torsion
#   group mode:
#   vdw/ffq

#   another interesting question then is what to do for the single option
#   this would essentially just correspond to running separate optimizations like the plain scipy optimizer
#   vectorization could speed this up for multiple systems, but then the optimizer itself has to be leaned down and vmapped
#   it's probably best to start with ffq group -> vdw group -> single mode bonded -> group mode bonded

#   should there also be a third ensemble mode that could take multiple copies of a perturbed structure???
#   could also change names to "bespoke" for single system optimization
#   and something like "ffield" for group
#   also add checking for invalid combinations
#   reax should work with both single and group but in principle it should probably throw an error for single
#   amber works for both
#   '''

# # dev_count = jax.device_count()
# # os_cpu_count = os.cpu_count()
# # num_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
# # num_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
# # total_allocated_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", "1"))
# # print("jax device count", dev_count)
# # print("os cpu count", os_cpu_count)
# # print("num_cpus_per_task", num_cpus_per_task)
# # print("num_tasks", num_tasks)
# # print("total_allocated_cpus", total_allocated_cpus)
# # # jax device count 1
# # # os cpu count 64
# # # num_cpus_per_task 2
# # # num_tasks 5
# # # total_allocated_cpus 10
# # sys.exit()




#     # for field in fields(new_ff):
#     #     attr = getattr(new_ff, field.name)
#     #     #if isinstance(attr, jnp.ndarray):
#     #     if field.name == "params_to_indices":
#     #         print(field.name, len(attr))
#     #     elif field.name == "solute_cut":
#     #         print(field.name, attr)
#     #     else:
#     #         print(field.name, attr.shape)



# #print("Process PID / affinity mask:", os.getpid(), os.sched_getaffinity(0))
# # print("Python executable:", sys.executable)
# # print("PATH:", os.environ["PATH"])
# # print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "not set"))


#   ###############################################################################
#   # #quick energy test
#   # from jax_md import minimize, space, simulate
#   # #print(systems[0])
#   # pos = systems[0].positions/10
#   # ff = force_fields[0]

#   # disp_fn, shift_fn = space.free()

#   # nrg_fn, amber_ff, body_fn, state = amber_energy(ff=ff, nonbonded_method="NoCutoff",
#   #                                             charge_method="FFQ", ensemble=None,
#   #                                             timestep=1e-3, init_temp=1e-3, return_charges=False, ffq_ff=ffq_ff, backprop_solve=True)

#   # nrg_g = jax.value_and_grad(nrg_fn)
#   # energy, grad = nrg_g(pos, ff, None)
#   # jax.debug.print("nrg pre {nrg}", nrg=energy)

#   # init_fn, apply_fn = minimize.gradient_descent(nrg_fn, shift_fn, 1e-6)
#   # # init_fn, apply_fn = simulate.nve(nrg_fn, shift_fn, 1e-3)

#   # state = init_fn(pos, ff=ff, nbr_list=[])
#   # #state = init_fn(jax.random.PRNGKey(0), pos, mass=ff.masses, kT=1e-5, ff=ff, nbr_list=[])

#   # def body_fn(i, state):
#   #   state, ff, _ = state
#   #   #jax.debug.print("pos {pos}", pos=pos)
#   #   state = apply_fn(state, ff=ff, nbr_list=None)

#   #   return state, ff, _

#   # new_state, amber_ff, nbr_list = jax.lax.fori_loop(0, 1000, body_fn, (state, ff, []))

#   # #energy = nrg_fn(new_state.position, ff, None)
#   # energy = nrg_fn(new_state, ff, None)
#   # jax.debug.print("nrg post {nrg}", nrg=energy)

#   # # import openmm.app as app
#   # # import openmm as omm
#   # # #inpcrd = app.AmberInpcrdFile(inpcrdFile)
#   # # prmtop = app.AmberPrmtopFile(f_list[0])
#   # # system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, nonbondedCutoff=.8*omm.unit.nanometer, removeCMMotion=False, rigidWater=False)
#   # # for i, f in enumerate(system.getForces()):
#   # #   f.setForceGroup(i)
#   # #   print(f)
#   # # platform = omm.Platform.getPlatformByName('CUDA')
#   # # properties = {'Precision': 'double'}
#   # # integrator = omm.VerletIntegrator(timestep*omm.unit.picoseconds)
#   # # simulation = omm.app.Simulation(prmtop.topology, system, integrator, platform=platform, platformProperties=properties)
#   # # simulation.context.setPositions(pos)

#   # # frcsum = 0
#   # # for i, f in enumerate(system.getForces()):
#   # #     state = simulation.context.getState(getEnergy=True, groups={i})
#   # #     print(f.getName(), (state.getPotentialEnergy()._value)/4.184)
#   # #     #print(f.getName(), "Uses PBCs", f.usesPeriodicBoundaryConditions())
#   # #     frcsum = frcsum + state.getPotentialEnergy()._value/4.184
#   # # print("OpenMM ", "OverallForce", ": ", frcsum, sep="")

#   # sys.exit()


#   #############################################################################
#   ### dlfind test

#   # from libdlfind import dl_find
#   # from libdlfind.callback import (dlf_get_gradient_wrapper,
#   #                                 dlf_put_coords_wrapper, make_dlf_get_params)
#   # import functools

#   # min_start = time.time()

#   # @dlf_get_gradient_wrapper
#   # def e_g_func(coordinates, iimage, kiter, ff, ffq_ff, nrg_fn):
#   #   energy, grad = nrg_fn(coordinates, ff, None)
#   #   return energy, grad

#   # @dlf_put_coords_wrapper
#   # def store_results(switch, energy, coordinates, iam, traj_coords, traj_energies):
#   #   traj_coords.append(onp.array(coordinates))
#   #   traj_energies.append(energy)
#   #   return

#   # pos = systems[0].positions/10
#   # ff = force_fields[0]

#   # nrg_fn, amber_ff, body_fn, state = amber_energy(ff=ff, nonbonded_method="NoCutoff",
#   #                                           charge_method="FFQ", ensemble=None,
#   #                                           timestep=1e-3, init_temp=1e-3, return_charges=False, ffq_ff=ffq_ff, backprop_solve=False)

#   # nrg_g = jax.value_and_grad(jax.jit(nrg_fn))

#   # traj_energies = []
#   # traj_coordinates = []

#   # dlf_get_params = make_dlf_get_params(coords=pos, maxcycle=1000)
#   # dlf_get_gradient = functools.partial(e_g_func, ff=ff, ffq_ff=None, nrg_fn=nrg_g)
#   # dlf_put_coords = functools.partial(
#   #     store_results, traj_coords=traj_coordinates, traj_energies=traj_energies
#   # )

#   # dl_find(
#   #       nvarin=len(pos) * 3,
#   #       dlf_get_gradient=dlf_get_gradient,
#   #       dlf_get_params=dlf_get_params,
#   #       dlf_put_coords=dlf_put_coords,
#   # )


#   # energy, grad = nrg_g(traj_coordinates[0], ff, None)
#   # print("RMS forces before minimization", jnp.sqrt(jnp.mean(jnp.sum((grad/4.184)**2, axis=1))))

#   # energy, grad = nrg_g(traj_coordinates[-1], ff, None)
#   # print("RMS forces after minimization", jnp.sqrt(jnp.mean(jnp.sum((grad/4.184)**2, axis=1))))

#   # print(f"Number of iterations: {len(traj_energies)}")
#   # print(f"Final energy (a.u.): {traj_energies[-1]}")

#   # min_end = time.time()

#   # print("min time", min_end-min_start)

#   # sys.exit()
  
#   ###########################################################################

#   ### torsion versions, some working, some not
#      # b1, b2, b3 = p2v - p1v, p3v - p2v, p4v - p3v
#     # b1 = jnp.mod(b1 + box * jnp.float64(0.5), box) - jnp.float64(0.5) * box
#     # b2 = jnp.mod(b2 + box * jnp.float64(0.5), box) - jnp.float64(0.5) * box
#     # b3 = jnp.mod(b3 + box * jnp.float64(0.5), box) - jnp.float64(0.5) * box

#     # c1 = jnp.cross(b2, b3)
#     # c2 = jnp.cross(b1, b2)

#     # p1 = (b1 * c1).sum(-1)
#     # p1 = p1 * safe_sqrt((b2 * b2).sum(-1))
#     # p2 = (c1 * c2).sum(-1)
#     # p2 = jnp.where(jnp.isclose(p2, 0.), 1, p2)

#     # r = jnp.arctan2(p1, p2)

#     ############################

#     # d12 = p2v-p1v
#     # d23 = p3v-p2v
#     # d34 = p4v-p3v

#     # n1 = jnp.cross(d12, d23)
#     # n2 = jnp.cross(d23, d34)
    
#     # cos_angle = jnp.dot(n1, n2) / (jnp.linalg.norm(n1) * jnp.linalg.norm(n2))

#     # r = jnp.arccos(cos_angle)

#     ############################

#     # # TODO be careful about this, these aren't periodic displacements
#     # b1, b2, b3 = p2v - p1v, p3v - p2v, p4v - p3v

#     # c1 = jnp.cross(b2, b3)
#     # c2 = jnp.cross(b1, b2)

#     # p1 = (b1 * c1).sum(-1)
#     # p1 = p1 * safe_sqrt((b2 * b2).sum(-1))
#     # p2 = (c1 * c2).sum(-1)
#     # p2 = jnp.where(jnp.isclose(p2, 0.), 1, p2)

#     # r = jnp.arctan2(p1, p2)

#     # # print("original r", r)

#     ############################

#     # d1 = p2v-p1v
#     # d2 = p2v-p3v
#     # d3 = p4v-p3v

#     # epsilon = 1e-7
#     # normal_1 = jnp.cross(d1, d2)
#     # normal_2 = jnp.cross(d2, d3)
#     # normal_plane_cross = jnp.cross(normal_1, normal_2)
#     # dr_32 = space.square_distance(d2)
#     # safe_dr_32 = util.safe_mask(dr_32 > 0, jnp.sqrt, dr_32, epsilon)
#     # x1 = jnp.sum(jnp.multiply(normal_plane_cross, d2 / safe_dr_32), axis=-1)
#     # x2 = jnp.sum(jnp.multiply(normal_1, normal_2), axis=-1)
#     # r = jnp.arctan2(x1, x2)

#     #############################

#     # Praxeolitic formula - 1 sqrt, 1 cross

#     # needs to be lifted to d1 = disp_fn(p1v,p2v)
#     # d1 = p1v-p2v
#     # d2 = p3v-p2v
#     # d3 = p4v-p3v

#     # d2n = d2/jnp.linalg.norm(d2)

#     # v = d1 - jnp.dot(d1, d2)*d2n
#     # w = d3 - jnp.dot(d3, d2)*d2n

#     # x = jnp.dot(v, w)
#     # y = jnp.dot(jnp.cross(d2n, v), w)
#     # r = jnp.arctan2(y, x)

#     # print("new r", r)

#     # # sys.exit()

#     ##########################################

# # NOTES FOR TORSIONS ABOVE
# # TODO there is probably a better way of doing these
# # precomputing distances for everything in the neighbor list and then just pulling these indices out
# # also redoing the masking so that you don't need to have box vectors at all
# # various torsion schemes that range -pi to pi, 0 to pi, etc, which is most correct?
# # is this the purpose of phase?
# # also consider if angles or norm'd vectors can be used from angle calculation

# # ANGLE COMMENTS
# # may need to do jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
# # TODO see if using clip works instead of this, not clear which one is more efficient
# # also look at cagri's safe sqrt example plus sam's comment in jax issue 1052, custom jvp is even more efficient
# #linalg norm isn't safe if the vector is 0, this doesn't seem like a great approach though
# #this is essentially what sam and cagri did though
# # at the very least using disp and dist function from setup would be better
# # calculates single angles and torsions, vmapped over list
# # also remap both of these to just take displacements eg. angle(d12, d23)


# # TODO reformat everything to 2 space tabs and 80? 99? 120? character line limit
# # TODO make sure lower is correct when indexing into gafftypes, solvent atoms in some prmtop are upper case
# # while ffq params can vary

# # class CustomEnumMeta(EnumMeta):
# #     def __getitem__(cls, name):
# #         # Custom mapping for special cases
# #         lookup = {"n+": "n_"}
# #         name = lookup.get(name, name)  # Translate "n+" to "n_" if needed
# #         return super().__getitem__(name)

# # TODO look into jnp.take and indexing
# # TODO change all function calls to use keyword args where applicable for readability and debugging

# # take inspiration from jax pme, dmff, mm commit, openmm/sander py interface,
# # move this to sparta git at some point
# # also need to implement free energy, shake, hmr
# # general rules should be to follow jax md format, the other mm example, dmff, and openmm
# # flags are needed for periodic/non periodic as well as setting up the associated structures
# # should probably do this reax style where inpcrd+prmtop -> struct
# # struct: pos, box, prms, etc - fully vectorizable jax dataclass
# # also need to implement basic charmm/gromacs terms and add switches for all 3
# # could also consider implementing a martini or go forcefield and parsers
# # full interoperability with openmm system or topology objects would also be desireable too
# # also need to figure out non orthogonal boxes. will this just work with jax md?
# # how do angle/torsion calculations work?
# # add a charges parameter to the overall energy function so that dynamic charges can be passed in as needed
# # also need to add helper functions to enable nvt/npt as well as shake, hmr, etc
# # add genesis or amber style control file and full md engine
# # start with fully lean implementation including pme and establish firm performance benchmarks
# # also add single/mixed/double precision flag , i wonder if just doing this in the ff dataclass will propogate correctly
# # port all of sams pme code in here and delete all changes to the original
# # think more about units and determine how you want to store everything internally and also what interface you want to provide
# # could you also implement custom jax type to mimic openmm quantity?
# # conditionals are fine if they pass through both functions and are decidable at compile time






#   #this may not be very efficient and is also likely out of place
#         #it really does warrant asking in this case if it will just be cheaper to eat the hit
#         #computing a corrective term for the direct and lj interactions using the exclusions list
#         #this probably ends up then scaling at least twice as much in memory
#         #but it does warrant asking if this is in place because of the entire replacement vs slice update
#         #another option might be jax.numpy.searchsorted
#         #where(isin) and setdiff are also options but some of these don't maintain ordering idx -> idx_after
#         #this just generally seems like a really difficult operation to express in jit
#         #mapped_setdiff1d = jax.vmap(jnp.setdiff1d)
#         #mapped_setdiff1d = jax.vmap(lambda a,b,c,d,e: jnp.setdiff1d(a,b,c, size=d, fill_value=e), in_axes=(0,0,None,None,None))

#         def mask_function(idx):
#             # TODO test assumed unique here, not clear from the algorithm if the junk fill values matter
#             # idx = mapped_setdiff1d(idx, exclusions_dense, False, idx.shape[1], ff.atom_count)


#         # TODO compare this to cagri's approach - it seems like this is expensive and memory consuming
#         # i want to say prmtops store exclusions differently anyways, it might be worth looking at this
#         #is this jitted?
#         # def mask_function(idx):
#         #     e_idx = jnp.argwhere(idx[ff.exclusions[:, 0]] == ff.exclusions[:, 1].reshape(-1,1), size=len(ff.exclusions), fill_value=ff.atom_count)
#         #     idx = idx.at[ff.exclusions[:, 0][e_idx[:, 0]], e_idx[:, 1]].set(ff.atom_count)
#         #     e_idx = jnp.argwhere(idx[ff.exclusions[:, 1]] == ff.exclusions[:, 0].reshape(-1,1), size=len(ff.exclusions), fill_value=ff.atom_count)
#         #     idx = idx.at[ff.exclusions[:, 1][e_idx[:, 0]], e_idx[:, 1]].set(ff.atom_count)

#         #     return idx

        # TODO make sure spacing after =, comma, and other things is consistent
        # TODO eventually do this
        #energy_fn = smap.pair_neighbor_list(masked_direct_fn)

        # if charge_method == "FFQ":
        #     # #TODO is this necessary?
        #     # atom_mask = ff.species >= 0
        #     # atm_mask = jnp.arange(len(ff.species))
        #     # atm_mask = atm_mask < ff.solute_cut
        #     # #print(atm_mask, atom_mask)
        #     # #sys.exit()
        #     # atom_mask = atom_mask * atm_mask
        #     # #atom_mask = atom_mask.at[ff.solute_cut:].set(0) # in the case where you have cutoff zero, i'm not sure if this should be done
        #     # # atom_mask = jnp.ones((ff.solute_cut,), dtype=jnp.int32) # only for no MM case
        #     # #atom_mask = 1
        #     atm_mask = 1 # TODO placeholder
            
        #     # # far_nbr_inds = jnp.tile(jnp.arange(ff.solute_cut), (ff.solute_cut, 1)) # only for no MM case
        #     # # far_nbr_inds = jnp.fill_diagonal(far_nbr_inds, ff.solute_cut, inplace=False) # only for no MM case
        #     # nbr_inds = jnp.tile(jnp.arange(len(ff.masses)), (len(ff.masses), 1))
        #     # nbr_inds = jnp.fill_diagonal(nbr_inds, ff.atom_count, inplace=False)
        #     # # print(far_nbr_inds)
        #     # # print(far_nbr_inds.shape)
        #     # # sys.exit()
        # else:
        #     charges = ff.charges
        #     charges_14 = ff.charges_14
        #     atom_mask = 1

# TODO figure out if some of these can be computed on the cpu and how independent different parameters are
# does this work with dataclasses split across devices? shard mapping?
# maybe do bonded terms on cpu and then just do pme on gpu?

# might also help to develop a filtration class and store number of bonds, angles, torsions, etc
# having a unified masking approach and function mapping like this will probably help
# with vmapping this function or extending to parameter optimization




# far_nbr_inds = jnp.tile(jnp.arange(ff.solute_cut), (ff.solute_cut, 1)) # only for no MM case
# far_nbr_inds = jnp.fill_diagonal(far_nbr_inds, ff.solute_cut, inplace=False) # only for no MM case

# select all indices where nbr is greater than ffq cutoff
# and then select those pairs where the first index is also outside the ffq index cutoff
# this ensures that only non_ffq, non_ffq are pruned while ffq, non_ffq or vice versa stay
# TODO this might not be necessary, but do consider if there's any use in precomputing a mask here

# TODO use jax.debug.callback and logging to dump this as well as some of the other data structures out to double precision
# nbr_mask = jnp.argwhere(far_nbr_inds >= ff.solute_cut)
# nbr_mask = nbr_mask[nbr_mask[:,0] >= ffq_cut]
# far_nbr_inds = far_nbr_inds.at[nbr_mask[:,0], nbr_mask[:,1]].set(ff.atom_count)
# far_nbr_dists = ffq_dist_fn(positions[:ff.solute_cut], R_far_nbr) # only for no MM case
#species = ff.species[:ff.solute_cut] # only for no MM case
# TODO implement more robust QM/MM framework

# far_nbr_mask = (far_nbr_inds != ff.solute_cut) & (atom_mask.reshape(-1,1) # only for no MM case
#                                         & atom_mask[far_nbr_inds])

# instead of ff.atom_count
# #ff.solute_cut, # only for no MM case

# if charge_method == "FFQMM":
#     sys.exit() # TODO not yet tested
#     chg_mask = jnp.arange(len(charges))
#     chg_mask = (chg_mask > ff.solute_cut) & (chg_mask < (len(charges)-1))
#     charges = jnp.where(chg_mask[:-1], ff.charges, charges[:-1])
#     charges_14 = charges[ff.pairs_14[:, 0]] * charges[ff.pairs_14[:, 1]]
# else:
#     charges_14 = charges[ff.pairs_14[:, 0]] * charges[ff.pairs_14[:, 1]]

        # angle_theta = jnp.where(angle_idx[:, 0] < 0, # doesn't work
        #                         0,
        #                         angle_fn(disp_map(a_pos[:, 0], a_pos[:, 1]), disp_map(a_pos[:, 2], a_pos[:, 1])))

# TODO also should positions/nblist be organized like this?
# i'm not sure which axis is fastest in jax
#nb_dist = jnp.linalg.norm(positions[nbr_list.idx[0, :]] - positions[nbr_list.idx[1, :]])

# TODO calculate distances while doing nb list instead of here
# just generally check this because it looks like the nb list gets distances anyways
# so why not save it for efficiency?
# also how to profile device memory with jax/sys/nsight

# TODO is this efficient? or do you need to define other variables to prevent constant copy semantics
# this is also probably somewhere i should maintain an array of precomputed pairs to avoid combining every time
# sigma = 0.5*(ff.sigma[ff.pairs[:, 0]] + ff.sigma[ff.pairs[:, 1]])


#TODO for nan gradients, consider if masking things like angle and torsion is the right approach
# TODO just generally do another pass for every division, arccos, sqrt, etc and add masking
# or if maybe safe masking the functions themselves is a better approach

# TODO if this ends up being expensive every time i could just do this every output_freq
# and then only pass raw nblists
# this might actually not be so slow
#ff = dataclasses.replace(ff, nbr_list=ff.nbr_list.update(state.position))
# nbr_list = nbr_list.update(state.position)
# state = apply_fn(state, ff=ff, nbr_list=nbr_list)

# look into more ways of doing this and also consider if sums are needed for mapped fns
# also should plain constants be wrapped in f32() i.e. f32(0.0), sam does this a lot
# fn = lambda dr: jnp.sum(4.0*epsilon*(idr12-idr6))
# return util.safe_mask(jnp.isclose(dr, 0.0), fn, dr, 0.0)

# print("lj", jnp.sum(4.0*epsilon*(idr12-idr6)))
# return jnp.sum(4.0*epsilon*(idr12-idr6))

# TODO change all tabs to either 2 or 4 standard in all of these files