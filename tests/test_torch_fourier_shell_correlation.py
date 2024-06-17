import torch

from torch_fourier_shell_correlation import fsc


def test_fsc():
    a = torch.zeros((10, 10, 10))
    a[4:6, 4:6, 4:6] = 1

    b = a
    result = fsc(a, b)
    assert torch.allclose(result, torch.ones(6))

    b = torch.rand((10, 10, 10))
    result = fsc(a, b)
    assert not torch.allclose(result, torch.ones(6))
