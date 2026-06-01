"""Common backend loading interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from jaxparamopt.config import OptimizationConfig

if TYPE_CHECKING:
  from jaxparamopt.input import InputBundle


class BackendInputLoader(Protocol):
  """Protocol for backend-specific input-loading adapters."""

  name: str

  def load_inputs(self, config: OptimizationConfig) -> "InputBundle":
    """Load backend-specific model, system, and parameter inputs."""
