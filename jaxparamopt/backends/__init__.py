"""Backend registry helpers."""

from __future__ import annotations

from .amber import AmberInputLoader
from .reaxff import ReaxFFInputLoader

_INPUT_LOADERS = {
  "amber": AmberInputLoader(),
  "reaxff": ReaxFFInputLoader(),
}


def get_backend_input_loader(name: str):
  try:
    return _INPUT_LOADERS[name]
  except KeyError as exc:
    known = ", ".join(sorted(_INPUT_LOADERS))
    raise ValueError(f"Unknown backend '{name}'. Known backends: {known}.") from exc
