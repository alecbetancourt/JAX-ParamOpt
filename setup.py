from setuptools import setup, find_packages
import io
import os
import subprocess
import re
import sys
'''
def get_cuda_version():
  try:
    result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE)
    out_str = result.stdout.decode('utf-8')
    regex = r'release (\S+),'
    match = re.search(regex, out_str)
    if match:
      return str(match.group(1))
    else:
      print("nvcc output cannot be parsed to receive the CUDA version")
      return None
  except:
    print("nvcc command cannot be run to find the CUDA version")
    return None


cuda_version = get_cuda_version()
if cuda_version == None:
  print("First CUDA needs to be installed")
  sys.exit(1) # exit on failure

print("Detected cuda version: ", cuda_version)

cuda_version = "cuda{}".format(cuda_version.replace(".",""))
#TODO: Automate installation for cuda dependent jaxlib
'''

'''
  'jax>=0.4.26, <=0.4.30',
  'jaxlib>=0.4.26,<=0.4.30',
  'jax_md @ git+https://github.com/cagrikymk/jax-md.git@reaxff_dev#egg=jax_md',
  'scipy>=1.2.1,<=1.12.0',
  'numpy',
  'numba>=0.56', # this needs to be figured out at some point, numba only supports old numpy version
  # as a note here, ideally pin these to tight version ranges and implement CI testing for a few different versions at a time
  # e.g. universal cuda 11 config, universal cuda 12 config
'''

INSTALL_REQUIRES = [
  'jax[cuda12]',
  'jaxlib',
  'scipy',
  'tabulate>=0.8.9',
  'frozendict',
  'tqdm',
  'optax',
  'jax_md @ git+https://github.com/alecbetancourt/jax-md.git@amber#egg=jax_md',
  'numpy<2.3.0', # parmed is broken with numpy >=2.3.0
  'numba',
  'evosax',
  'parmed',
  'h5py',
  'libdlfind',
  'openmm'
]

# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()
  
setup(
   name='jaxparamopt',
   version='0.1.0',
   author='William Betancourt',
   author_email='alecbetancourt@gmail.com',
   packages=["jaxparamopt"],
   entry_points={'console_scripts': ['jaxparamopt=jaxparamopt.driver:main']
                },
   url='https://github.com/alecbetancourt/JAX-ParamOpt',
   license='LICENSE',
   description='A gradient based framework for force field parameter optimization',
   long_description=long_description,
   long_description_content_type='text/markdown',
   python_requires='>=3.7',
   install_requires=INSTALL_REQUIRES,
   
)
