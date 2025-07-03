from typing import Sequence, Tuple

import einops
import torch

from torch_grid_utils import fftfreq_grid


def fsc(
    a: torch.Tensor,
    b: torch.Tensor,
    rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring/shell correlation between two square/cubic images."""
    # input handling
    image_shape = a.shape
    dft_shape = rfft_shape(image_shape)
    if a.ndim not in (2, 3):
        raise ValueError('images must be 2D or 3D.')
    elif a.shape != b.shape:
        raise ValueError('images must be the same shape.')
    elif rfft_mask is not None and rfft_mask.shape != dft_shape:
        raise ValueError('valid rfft indices must have same shape as rfft.')

    # linearise data and fftfreq of each component
    a, b = torch.fft.rfftn(a), torch.fft.rfftn(b)
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
        norm=True,
        device=a.device,
    )
    if rfft_mask is not None:
        a, b, frequencies = (arr[rfft_mask] for arr in [a, b, frequency_grid])
    else:
        a, b, frequencies = (torch.flatten(arg) for arg in [a, b, frequency_grid])

    # define frequency bins
    bin_centers = torch.fft.rfftfreq(image_shape[0], device=a.device)
    df = 1 / image_shape[0]

    # setup to split data at midpoint between frequency bin centers
    bin_centers = torch.cat([bin_centers, torch.as_tensor([0.5 + df], device=a.device)])
    bin_centers = bin_centers.unfold(dimension=0, size=2, step=1)  # (n_shells, 2)
    split_points = einops.reduce(bin_centers, 'shells high_low -> shells', reduction='mean')
    

    # find indices of all components in each shell
    sorted_frequencies, sort_idx = torch.sort(frequencies, descending=False)
    split_idx = torch.searchsorted(sorted_frequencies, split_points)
    shell_idx = torch.tensor_split(sort_idx, split_idx)[:-1]

    # calculate normalised cross correlation in each shell
    fsc = [
        _normalised_cc_complex_1d(a[idx], b[idx])
        for idx in
        shell_idx[1:]
    ]
    fsc = [1.0] + fsc  # fix the 0 DC shell to 1.0
    return torch.real(torch.tensor(fsc))


def rfft_shape(image_shape: Sequence[int]) -> Tuple[int]:
    """Calculate the shape of an rfft on an input image of a given shape."""
    rfft_shape = list(image_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def _normalised_cc_complex_1d(a: torch.Tensor, b: torch.Tensor):
    correlation = torch.dot(a, torch.conj(b))
    return correlation / (torch.linalg.norm(a) * torch.linalg.norm(b))
