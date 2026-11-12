import pytest
import yaml

from rfdd import prepare
from rfdd.events import AlarmLogic


@pytest.fixture(scope="session")
def smoke_cfg() -> dict:
    with open("configs/smoke.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def smoke_days(smoke_cfg: dict) -> dict:
    return prepare.load(smoke_cfg)


@pytest.fixture(scope="session")
def logic() -> AlarmLogic:
    return AlarmLogic()
