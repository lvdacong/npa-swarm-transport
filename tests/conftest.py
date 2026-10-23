import pytest
import torch

@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)
