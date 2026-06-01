"""Configuration objects and CLI parsing for jaxparamopt.

TODO: Revisit file-based configuration support after the workflow and backend
interfaces stabilize. YAML/TOML config loading is a potential option.
TODO: Sanity check configuration options and help output
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from .smartformatter import SmartFormatter


def validate_init_type(value: str) -> str:
  """Validate trial initialization mode without importing evosax eagerly."""
  closed_choices = {
    "random",
    "educated",
    "fixed",
    "cmaes",
    "openes",
    "pgpe",
    "snes",
    "lga",
    "diffev",
    "shgo",
    "direct",
    "basin",
    "sobol",
    "adaptive",
  }
  if value in closed_choices:
    return value

  if value.startswith("genetic_"):
    function_name = value[len("genetic_") :]
    try:
      import evosax
    except ImportError as exc:
      raise argparse.ArgumentTypeError(
        "genetic_* initialization requires the optional 'global-opt' dependencies."
      ) from exc

    if hasattr(evosax.algorithms, function_name):
      return value
    raise argparse.ArgumentTypeError(
      f"[ERROR] Function '{function_name}' not found in 'evosax.algorithms' module."
    )

  raise argparse.ArgumentTypeError(
    f"[ERROR] Invalid input: '{value}'. Must be one of the supported "
    "closed options or start with 'genetic_'."
  )


@dataclass(slots=True)
class OptimizationConfig:
  init_FF: str = "ffield"
  params: str = "params"
  geo: str = "geo"
  train_file: str = "trainset.in"
  use_valid: bool = False
  valid_file: str = "validset.in"
  valid_geo_file: str = "valid_geo"
  opt_method: str = "L-BFGS-B"
  num_trials: int = 1
  num_steps: int = 5
  loss_metric: str = "sse"
  init_FF_type: str = "fixed"
  random_sample_count: int = 0
  num_e_minim_steps: int = 0
  e_minim_LR: float = 5e-4
  end_RMSG: float = 1.0
  out_folder: str = "outputs"
  save_opt: str = "best"
  bonded_cutoff: float = 5.0
  cutoff2: float = 0.001
  max_num_clusters: int = 10
  perc_noise_when_stuck: float = 0.04
  seed: int = 0
  ff_type: str = "reaxff"
  ffq_params: str = "ffq_params"
  opt_mode: str = "group"
  amber_pme: bool = False
  amber_charge: str = "GAFF"
  min_type: str = "grad"
  relative_energies: bool = False
  debug_level: int = 0

  @classmethod
  def from_namespace(cls, namespace: argparse.Namespace) -> "OptimizationConfig":
    return cls(**vars(namespace))

  @classmethod
  def from_mapping(cls, values: Mapping[str, Any]) -> "OptimizationConfig":
    base_values = asdict(cls())
    unknown = sorted(set(values) - set(base_values))
    if unknown:
      raise TypeError(f"Unknown configuration field(s): {', '.join(unknown)}")
    base_values.update(values)
    return cls(**base_values)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  def to_namespace(self) -> SimpleNamespace:
    return SimpleNamespace(**self.to_dict())


def build_argument_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="JAX-ParamOpt driver",
    formatter_class=SmartFormatter,
  )
  parser.add_argument(
    "--init_FF",
    metavar="filename",
    type=str,
    default="ffield",
    help="Initial force field file",
  )
  parser.add_argument(
    "--params",
    metavar="filename",
    type=str,
    default="params",
    help="Parameters file",
  )
  parser.add_argument(
    "--geo",
    metavar="filename",
    type=str,
    default="geo",
    help="Geometry file",
  )
  parser.add_argument(
    "--train_file",
    metavar="filename",
    type=str,
    default="trainset.in",
    help="Training set file",
  )
  parser.add_argument(
    "--use_valid",
    action="store_true",
    help="Enable validation data support.",
  )
  parser.add_argument(
    "--valid_file",
    metavar="filename",
    type=str,
    default="validset.in",
    help="Validation set file (same format as trainset.in)",
  )
  parser.add_argument(
    "--valid_geo_file",
    metavar="filename",
    type=str,
    default="valid_geo",
    help="Geo file for the validation data",
  )
  parser.add_argument(
    "--opt_method",
    metavar="method",
    choices=["L-BFGS-B", "SLSQP", "TNC", "trust-constr"],
    type=str,
    default="L-BFGS-B",
    help='Optimization method - "L-BFGS-B" or "SLSQP"',
  )
  parser.add_argument(
    "--num_trials",
    metavar="number",
    type=int,
    default=1,
    help="R|Number of trials (Population size).\n"
    "If set to <= 0, provided force field will be evaluated w/o any training (init_FF).",
  )
  parser.add_argument(
    "--num_steps",
    metavar="number",
    type=int,
    default=5,
    help="Number of optimization steps per trial",
  )
  parser.add_argument(
    "--loss_metric",
    metavar="method",
    choices=["sse", "mse", "rmse", "mae", "huber", "logcosh", "sum"],
    type=str,
    default="sse",
    help='Scoring metric to use in loss function, either "sse", "rmse"',
  )
  parser.add_argument(
    "--init_FF_type",
    metavar="init_type",
    type=validate_init_type,
    default="fixed",
    help="R|How to start the trials from the given initial force field.\n"
    '"random": Sample the parameters from uniform distribution between given ranges.\n'
    '"educated": Sample the parameters from a uniform distribution centered at given values.\n'
    '"fixed": Start from the parameters given in "init_FF" file',
  )
  parser.add_argument(
    "--random_sample_count",
    metavar="number",
    type=int,
    default=0,
    help="R|Before the optimization starts, uniforms sample the paramater space.\n"
    'Select the best sample to start the training with, only works with "random" inital start.\n'
    "if set to 0, no random search step will be skipped. ",
  )
  parser.add_argument(
    "--num_e_minim_steps",
    metavar="number",
    type=int,
    default=0,
    help="Number of energy minimization steps",
  )
  parser.add_argument(
    "--e_minim_LR",
    metavar="init_LR",
    type=float,
    default=5e-4,
    help="Initial learning rate for energy minimization",
  )
  parser.add_argument(
    "--end_RMSG",
    metavar="end_RMSG",
    type=float,
    default=1.0,
    help="Stopping condition for E. minimization",
  )
  parser.add_argument(
    "--out_folder",
    metavar="folder",
    type=str,
    default="outputs",
    help="Folder to store the output files",
  )
  parser.add_argument(
    "--save_opt",
    metavar="option",
    choices=["all", "best"],
    default="best",
    help='R|"all" or "best"\n'
    '"all": save all of the trained force fields\n'
    '"best": save only the best force field',
  )
  parser.add_argument(
    "--bonded_cutoff",
    metavar="cutoff",
    type=float,
    default=5.0,
    help="Cutoff distance for bonded interactions (in Angstrom).",
  )
  parser.add_argument(
    "--cutoff2",
    metavar="cutoff",
    type=float,
    default=0.001,
    help="BO-cutoff for valency angles and torsion angles",
  )
  parser.add_argument(
    "--max_num_clusters",
    metavar="max # clusters",
    type=int,
    default=10,
    choices=range(1, 16),
    help="R|Max number of clusters that can be used\n"
    "High number of clusters lowers the memory cost\n"
    "However, it increases compilation time,especially for cpus",
  )
  parser.add_argument(
    "--perc_noise_when_stuck",
    metavar="percentage",
    type=float,
    default=0.04,
    help="R|Percentage of the noise that will be added to the parameters\n"
    "when the optimizer is stuck.\n"
    "param_noise_i = (param_min_i, param_max_i) * perc_noise_when_stuck\n"
    "Small values such as 0 to 10% are generally recommended.",
  )
  parser.add_argument(
    "--seed",
    metavar="seed",
    type=int,
    default=0,
    help="Seed value",
  )
  parser.add_argument(
    "--ff_type",
    metavar="ff_type",
    choices=["reaxff", "amber", "ambereem"],
    type=str,
    default="reaxff",
    help='Forcefield to optimize - "reaxff" or "amber" or "ambereem"',
  )
  parser.add_argument(
    "--ffq_params",
    metavar="filename",
    type=str,
    default="ffq_params",
    help="Supplemental parameter file for FFQ if AMBER is enabled",
  )
  parser.add_argument(
    "--opt_mode",
    metavar="mode",
    choices=["single", "group"],
    type=str,
    default="group",
    help="For AMBER, determines optimization type\n"
    "single - optimization is for a single system\n"
    "group - optimization is done for a group of files that\n"
    "share common parameters defined in the params file",
  )
  parser.add_argument(
    "--amber_pme",
    action="store_true",
    help="Enable PME or related AMBER-specific electrostatics behavior when supported.",
  )
  parser.add_argument(
    "--amber_charge",
    metavar="charge_model",
    choices=["GAFF", "FFQ", "LRCH"],
    type=str,
    default="GAFF",
    help="Method to use for calculating AMBER charges\n"
    '"GAFF" uses default charges from .prmtop file'
    '"FFQ" uses charges generated with EEM'
    '"LRCH" uses a linear response based solver',
  )
  parser.add_argument(
    "--min_type",
    metavar="min_type",
    choices=["grad", "fire", "dlfind", "bfgs"],
    type=str,
    default="grad",
    help='Method to use for energy minimization - "grad" or "dlfind" or "bfgs"\n'
    '"dlfind" uses internal coordinates and calls libdlfind library'
    '"grad" uses gradient descent internally to optimize structures'
    '"bfgs" uses Scipy L-BFGS-B to optimize strucutres',
  )
  parser.add_argument(
    "--relative_energies",
    action="store_true",
    help="Use relative energies instead of absolute energies for loss evaluation.",
  )
  parser.add_argument(
    "--debug_level",
    metavar="level",
    type=int,
    default=0,
    choices=range(0, 10),
    help="The debug/reporting level that the optimizer will run\n"
    "Many of the higher levels use JAX callbacks that are slow\n"
    "0 - (default) No debugging information\n"
    "1 - More timing information printed per loop\n"
    "2 - Per component loss information\n"
    "3 - Per component, per structure loss information\n"
    "4 - Full loss gradient information\n"
    "5 - Full dump of internal optimizer state to hdf5 log",
  )
  return parser


def parse_cli_args(argv: list[str] | None = None) -> OptimizationConfig:
  parser = build_argument_parser()
  namespace = parser.parse_args(argv)
  return OptimizationConfig.from_namespace(namespace)
