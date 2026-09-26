import pytest

from finpilot.data.generate import build
from finpilot.db import make_engine


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    """A small, seeded bank database built once for the whole test run."""
    path = tmp_path_factory.mktemp("db") / "test.db"
    eng = make_engine(f"sqlite:///{path.as_posix()}")
    build(eng, n_users=5, days=90, seed=1)
    return eng
