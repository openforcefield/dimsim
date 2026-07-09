import parsl
import pytest
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor


@pytest.fixture(scope="session", autouse=True)
def parsl_test_setup():
    # Use local threads for predictable and fast unit testing
    local_config = Config(
        executors=[ThreadPoolExecutor(label="local_threads", max_threads=2)],
        strategy=None,  # Disable dynamic scaling during unit tests
    )
    parsl.load(local_config)

    yield  # Run tests

    parsl.clear()  # Teardown
