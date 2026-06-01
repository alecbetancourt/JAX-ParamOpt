"""AMBER-family backend input-loading placeholders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxparamopt.config import OptimizationConfig

if TYPE_CHECKING:
  from jaxparamopt.input import InputBundle


class AmberInputLoader:
  name = "amber"

  def load_inputs(self, config: OptimizationConfig) -> "InputBundle":
    raise NotImplementedError(
        "AMBER input loading has not been extracted from driver.py yet."
    )
