import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def params_path():
    return ROOT / "configs" / "params.yaml"


@pytest.fixture
def project_root():
    return ROOT
