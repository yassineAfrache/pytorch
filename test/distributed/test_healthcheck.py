import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)
from torch._C._distributed_c10d import (
    HealthcheckNCCL,
)

class HealthcheckTest(TestCase):
    def test_healthcheck_nccl(self) -> None:
        pass

if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
