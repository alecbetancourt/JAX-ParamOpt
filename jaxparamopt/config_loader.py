# support for yaml input files
# useful for scalability and easier configuration of the many options available

import argparse
import yaml
import os
from typing import Any, Dict

def load_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Optimization run configuration")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional override for random seed.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode.")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    config["debug"] = args.debug

    return config