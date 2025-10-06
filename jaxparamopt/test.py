# use absl testing as in jax md

from absl.testing import absltest
from absl.testing import parameterized

#from jax import random
import jax
#from jax import jit, vmap, grad
#from jax.tree_util import tree_map
import jax.numpy as jnp
#from scipy.io import loadmat

jax.config.parse_flags_with_absl()