import pytest

from jaxparamopt.config import OptimizationConfig
from jaxparamopt.input import InputBundle, ParameterInput, load_inputs


def test_load_inputs_dispatches_to_backend_loader(monkeypatch):
  config = OptimizationConfig(ff_type="reaxff", init_FF="ffield")

  class FakeLoader:
    def load_inputs(self, normalized_config):
      return InputBundle(
        config=normalized_config,
        backend_name=normalized_config.ff_type,
        parameter_input=ParameterInput(
          parameter_ids=("p1",),
          bounds=((0.0, 1.0),),
        ),
      )

  monkeypatch.setattr(
    "jaxparamopt.input.get_backend_input_loader",
    lambda name: FakeLoader(),
  )

  bundle = load_inputs(config)

  assert isinstance(bundle, InputBundle)
  assert bundle.backend_name == "reaxff"
  assert bundle.parameter_input.parameter_ids == ("p1",)


def test_load_inputs_accepts_mapping(monkeypatch):
  class FakeLoader:
    def load_inputs(self, normalized_config):
      return InputBundle(
        config=normalized_config,
        backend_name=normalized_config.ff_type,
      )

  monkeypatch.setattr(
    "jaxparamopt.input.get_backend_input_loader",
    lambda name: FakeLoader(),
  )

  bundle = load_inputs({"ff_type": "amber", "seed": 11})

  assert bundle.backend_name == "amber"
  assert bundle.config.seed == 11


def test_load_inputs_raises_for_unknown_backend():
  with pytest.raises(ValueError, match="Unknown backend"):
    load_inputs({"ff_type": "unknown-backend"})
