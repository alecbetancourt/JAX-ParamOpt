# module for global optimization algorithms

import sys
import jax
import jax.numpy as jnp

from jaxparamopt.optimizer import (add_noise_to_params, random_parameter_search)
import evosax
import scipy

# handler for all global optimization options
def global_optimization(init_params, bounds, random_sample_count, init_FF_type, loss_func, loss_and_grad_func, dtype, loss_args):
    if init_FF_type == 'random':
        min_params = random_parameter_search(bounds, random_sample_count, loss_func, dtype, loss_args)
        selected_params = min_params
    elif init_FF_type == 'educated':
        selected_params = add_noise_to_params(init_params, bounds, scale=0.1)
    elif init_FF_type == 'fixed': # fixed
        selected_params = jnp.array(init_params)
    elif init_FF_type == 'sobol':
        selected_params = sobol_opt(bounds, random_sample_count, loss_func, loss_args)
    elif init_FF_type == 'direct':
        selected_params = direct_opt(bounds, loss_func, loss_args)
    elif init_FF_type == 'lga':
        sys.exit("Init type not implemented")
    elif init_FF_type == 'diffev':
        selected_params = diffev_opt(bounds, loss_func, loss_args)
    elif init_FF_type == 'shgo': # needs grad function
        selected_params = shgo_opt(bounds, loss_and_grad_func, loss_args)
    elif init_FF_type == 'basin':
        sys.exit("Init type not implemented")
    elif init_FF_type == 'adaptive':
        sys.exit("Init type not implemented")
    elif init_FF_type.startswith("genetic_"):
        selected_params = genetic_opt(bounds, random_sample_count, init_FF_type, loss_func, loss_args)

    print("[INFO] Loss with initial parameter set is:", loss_func(init_params, *loss_args))
    if init_FF_type != 'fixed':
        print("[INFO] Loss with parameter set after global optimization is:", loss_func(selected_params, *loss_args))

    return selected_params

def sobol_opt(bounds, random_sample_count, loss_func, loss_args):
    #m for random_base2 is a power of 2
    m = int(jnp.log2(random_sample_count))
    print(f"[INFO] Random sample count is truncated from {random_sample_count} to nearest power of 2, {m}, for efficiency reasons")
    sampler = scipy.stats.qmc.Sobol(d=len(bounds), scramble=False)
    sample = sampler.random_base2(m=m)
    # scale random distribution to bounds
    # TODO does this preserve the properties of the sequence?
    sample = scipy.stats.qmc.scale(sample, bounds[:, 0], bounds[:, 1])

    print("sample dims", sample.shape, len(bounds), random_sample_count)

    #losses = jax.vmap(new_loss_func, in_axes=(0,None))(sample, *args_loss)
    losses = jnp.array([loss_func(p, *loss_args) for p in sample], dtype=jnp.float32)
    best_loss_idx = jnp.argmin(losses)
    best_loss = losses[best_loss_idx]
    best_params = sample[best_loss_idx]
    selected_params = best_params
    
    return selected_params
    # TODO what other relevant statistics to include for this?
    # mean, median, stdev, etc
    # lots of parameter guesses are completely unphysical so this is an interesting question

def direct_opt(bounds, loss_func, loss_args):
    #TODO this likely requires more tuning due to memory issues
    #direct_options = dict(maxiter=10, maxls=20, maxcor=20, disp=False, jac=True) # maxiter 100
    direct_min_options = dict(method='L-BFGS-B')
    opt_bounds = scipy.optimize.Bounds(bounds[:,0], bounds[:,1])
    opt_results = scipy.optimize.direct(loss_func, bounds=opt_bounds, args=loss_args,
                                #options=direct_options,
                                #minimizer_kwargs=direct_min_options
                                )

    selected_params = opt_results.x
    return selected_params

def dual_annealing_opt(bounds, loss_func, loss_args):
    da_options = dict(maxiter=1000, disp=False) # maxiter 100
    opt_bounds = scipy.optimize.Bounds(bounds[:,0], bounds[:,1])
    opt_results = scipy.optimize.dual_annealing(loss_func, bounds=opt_bounds, args=loss_args,
                                options=da_options
                                )

    selected_params = opt_results.x

    jax.debug.print("[INFO] Dual Annealing Optimization Results {}", opt_results)
    return selected_params

def diffev_opt(bounds, loss_func, loss_args):
    print("doing differential evolution opt")
    # TODO add control for number of iterations
    de_options = dict(maxiter=1000, disp=False) # maxiter 100
    opt_bounds = scipy.optimize.Bounds(bounds[:,0], bounds[:,1])
    opt_results = scipy.optimize.differential_evolution(loss_func, bounds=opt_bounds, args=loss_args,
                                options=de_options
                                )
    selected_params = opt_results.x
    jax.debug.print("[INFO] Differential Evolution Optimization Results {}", opt_results)
    return selected_params

def shgo_opt(bounds, loss_and_grad_func, loss_args):
    shgo_options = dict(maxiter=10, maxls=20, maxcor=20, disp=False, jac=True) # maxiter 100
    shgo_min_options = dict(method='L-BFGS-B')
    opt_results = scipy.optimize.shgo(loss_and_grad_func, bounds=bounds, args=loss_args,
                                options=shgo_options, minimizer_kwargs=shgo_min_options)

    selected_params = opt_results.x
    jax.debug.print("[INFO] SHGO Optimization Results {}", opt_results)
    return selected_params

def genetic_opt(bounds, random_sample_count, init_FF_type, loss_func, loss_args):
    rng = jax.random.PRNGKey(0)
    initialization = jax.random.uniform(rng, (len(bounds),), minval=bounds[:,0], maxval=bounds[:,1])

    strategy_name = init_FF_type[len("genetic_"):]
    print("[INFO] Genetic algorithm is being used, strategy:", strategy_name)
    strategy_fn = getattr(evosax.algorithms, strategy_name)
    strategy = strategy_fn(population_size=256, solution=initialization)

    es_params = strategy.default_params
    state = strategy.init(rng, initialization, es_params)

    print("[INFO] Init params with random guess", initialization)
    print("[INFO] Starting loss", loss_func(jnp.array(initialization), *loss_args))

    # Run ask-eval-tell loop - NOTE: By default minimization!
    fit_list = []
    for t in range(random_sample_count):
        #TODO: include vmap
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_gen, state, es_params)
        x = jnp.clip(x, bounds[:,0], bounds[:,1])
        fitness = jnp.array([loss_func(p, *loss_args) for p in x], dtype=jnp.float32)
        state, metrics = strategy.tell(rng_eval, x, fitness, state, es_params)
        if (t + 1) % 10 == 0:
            print("# Gen: {}|Fitness: {:.5f}".format(t+1, state.best_fitness))
    return state.best_solution
         

# if(args.init_FF_type in ['cmaes','snes','openes','pgpe']):
#   #rng = jax.random.PRNGKey(int(time.time()))
#   rng = jax.random.PRNGKey(0)
#   args_loss = (param_indices, force_field, training_data,
#       list_positions, aligned_data, center_sizes, args.ff_type, args.opt_mode, ffq_ff)
#   # es_params = strategy.default_params
#   # state = strategy.initialize(rng, es_params)
#   #TODO: consider replacing this with a random distribution
#   initialization = jax.random.uniform(rng, (len(bounds),), minval=bounds[:,0], maxval=bounds[:,1])

#   if args.init_FF_type == 'cmaes':
#     from evosax.algorithms import Sep_CMA_ES
#     strategy = Sep_CMA_ES(population_size=256, solution=initialization)
#   elif args.init_FF_type == 'snes':      
#     from evosax.algorithms import SNES
#     strategy = SNES(population_size=256, solution=initialization)
#   elif args.init_FF_type == 'openes':      
#     from evosax.algorithms import Open_ES
#     strategy = Open_ES(population_size=256, solution=initialization)
#   elif args.init_FF_type == 'pgpe':      
#     from evosax.algorithms import PGPE
#     strategy = PGPE(population_size=256, solution=initialization)
#   elif args.init_FF_type == 'lga':      
#     from evosax.algorithms import LGA
#     strategy = LGA(population_size=256, solution=initialization)
#   elif args.init_FF_type == 'diffev':      
#     from evosax.algorithms import DiffusionEvolution
#     strategy = DiffusionEvolution(population_size=256, solution=initialization)

#   es_params = strategy.default_params
#   state = strategy.init(rng, initialization, es_params)

#   #state = state.replace(best_member=jnp.array(initialization))
#   #state = state.replace(mean=jnp.array(initialization))
#   print("[INFO] Init Params", initialization)
#   print("[INFO] Starting loss", new_loss_func(jnp.array(initialization), *args_loss))
#   #l_f = jax.vmap(new_loss_func, in_axes=(0,None,None,None,None,None,None,None,None,None,None))
#   #                               out_axes=(0,None,None,None,None,None,None,None,None,None,None))

#   # Run ask-eval-tell loop - NOTE: By default minimization!
#   gen_start = time.time()
#   fit_list = []
#   for t in range(args.random_sample_count):
#     #TODO: include vmap
#     rng, rng_gen, rng_eval = jax.random.split(rng, 3)
#     x, state = strategy.ask(rng_gen, state, es_params)
#     x = jnp.clip(x, bounds[:,0], bounds[:,1])
#     fitness = jnp.array([new_loss_func(p, *args_loss) for p in x], dtype=jnp.float32)
#     state, metrics = strategy.tell(rng_eval, x, fitness, state, es_params)
#     if (t + 1) % 10 == 0:
#       print("# Gen: {}|Fitness: {:.5f}".format(t+1, state.best_fitness))

#   print("[INFO] Best Solution:", state.best_solution)
#   print("[INFO] Best Fitness:", state.best_fitness)
#   selected_params = state.best_solution
#   gen_end = time.time()
#   print("Genetic Optimization Time:", gen_end-gen_start)

# if args.init_FF_type == 'pcmaes':
#   sys.exit("[ERROR] Parallel GAs not fully implemented")
#   #TODO example implementation for
#   #parallel solver using shard map
#   from jax.sharding import Mesh
#   from jax.sharding import PartitionSpec
#   from jax.sharding import NamedSharding
#   from jax.experimental import mesh_utils
#   from jax.experimental.shard_map import shard_map

#   print("JAX Devices", jax.devices())
#   P = jax.sharding.PartitionSpec
#   devices = mesh_utils.create_device_mesh((4,))
#   mesh = jax.sharding.Mesh(devices, ('x'))
#   sharding = jax.sharding.NamedSharding(mesh, P('x'))

#   rng = jax.random.PRNGKey(0)
#   rng = jax.random.split(rng, 4)
#   args_loss = (param_indices, force_field, training_data,
#       list_positions, aligned_data, center_sizes, False,
#       aligned_amber_ff, ff_type_int, charge_type_int)
#   es_params = strategy.default_params
#   es_params = es_params.replace(mu_eff=jnp.repeat(es_params.mu_eff, 4))
#   es_params = es_params.replace(c_1=jnp.repeat(es_params.c_1, 4))
#   es_params = es_params.replace(c_mu=jnp.repeat(es_params.c_mu, 4))
#   es_params = es_params.replace(c_sigma=jnp.repeat(es_params.c_sigma, 4))
#   es_params = es_params.replace(d_sigma=jnp.repeat(es_params.d_sigma, 4))
#   es_params = es_params.replace(c_c=jnp.repeat(es_params.c_c, 4))
#   es_params = es_params.replace(chi_n=jnp.repeat(es_params.chi_n, 4))
#   es_params = es_params.replace(c_m=jnp.repeat(es_params.c_m, 4))
#   es_params = es_params.replace(sigma_init=jnp.repeat(es_params.sigma_init, 4))

#   es_params = es_params.replace(init_min=jnp.broadcast_to(bounds[:,0],(4,)+bounds[:,0].shape))
#   es_params = es_params.replace(init_max=jnp.broadcast_to(bounds[:,1],(4,)+bounds[:,1].shape))
#   es_params = es_params.replace(clip_min=jnp.broadcast_to(bounds[:,0],(4,)+bounds[:,0].shape))
#   es_params = es_params.replace(clip_max=jnp.broadcast_to(bounds[:,1],(4,)+bounds[:,1].shape))

#   #TODO probably a better way to do this by iterating over the fields and doing replace(**kwargs)
#   #for field in es_params.fields

#   #init_fn = shard_map(strategy.initialize, mesh=mesh, in_specs=(P(None), P("x")), out_specs=P("x"))
#   init_fn = jax.jit(jax.vmap(strategy.initialize))
#   print("clip shape", es_params.clip_max.shape)
#   print("mu shape", es_params.c_mu.shape)
#   es_params_sharded = jax.device_put(es_params, sharding)
#   rng_sharded = jax.device_put(rng, sharding)

#   #jax.debug.visualize_array_sharding(es_params_sharded)
#   #print("es sharded devices", es_params_sharded.init_min.devices())

#   state = init_fn(rng_sharded, es_params_sharded)
#   #state = init_fn(rng, es_params)

#   print("PCMAES Best Member", state.best_member.shape)

#   gen_start = time.time()

#   #TODO: better to do jit of shard map than shard map of jit if i go that direction
#   @jax.jit
#   @jax.vmap
#   def opt_loop(rng, es_params, state):
#     jax.debug.print("Optimization loop starting")
#     for t in range(args.generations):
#       #TODO: include vmap
#       rng, rng_gen, rng_eval = jax.random.split(rng, 3)
#       x, state = strategy.ask(rng_gen, state, es_params)
#       fitness = jnp.array([new_loss_func(p, *args_loss) for p in x], dtype=jnp.float32)
#       state = strategy.tell(x, fitness, state, es_params)
#       if (t + 1) % 1 == 0:
#         jax.debug.print("# Gen: {gen}", gen=t+1)
#         sys.stdout.flush()

#     return state

#   state = opt_loop(rng_sharded, es_params_sharded, state)

#   print("Best Member:", state.best_member)
#   print("Best Fitness:", state.best_fitness)
#   selected_params = state.best_member
#   gen_end = time.time()
#   print("Genetic Optimization Time:", gen_end-gen_start)

#   # TODO try to gather at end with lax.all_gather?
#   sys.exit()