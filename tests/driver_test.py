import pytest

from jaxparamopt import driver


def test_main_exits_cleanly_after_runtime_setup():
  with pytest.raises(SystemExit) as exc_info:
    driver.main([])

  assert exc_info.value.code == 0
