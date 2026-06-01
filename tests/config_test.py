from jaxparamopt.config import OptimizationConfig, parse_cli_args


def test_parse_cli_args_builds_typed_config():
  config = parse_cli_args(
      [
          "--init_FF",
          "ffield_custom",
          "--use_valid",
          "--num_trials",
          "3",
      ]
  )

  assert isinstance(config, OptimizationConfig)
  assert config.init_FF == "ffield_custom"
  assert config.use_valid is True
  assert config.num_trials == 3


def test_config_from_mapping_overrides_defaults():
  config = OptimizationConfig.from_mapping(
      {
          "seed": 7,
          "ff_type": "amber",
      }
  )

  assert isinstance(config, OptimizationConfig)
  assert config.seed == 7
  assert config.ff_type == "amber"
  assert config.num_trials == 1
