"""Fourier shell correlation between two 2D or 3D images."""

from typing import Sequence, Tuple

import einops
import torch
from torch_grid_utils import fftfreq_grid


def fourier_ring_correlation(
    a: torch.Tensor, b: torch.Tensor, rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring correlation between two 2D images with batching support.

    Args:
        a: Input tensor of shape (..., h, w)
        b: Input tensor of shape (..., h, w)
        rfft_mask: Optional mask for rfft, shape should match rfft output

    Returns
    -------
        Correlation values of shape (..., min(h, w) // 2 + 1)
    """
    return _fourier_correlation(a, b, rfft_mask, ndim=2)


def fourier_shell_correlation(
    a: torch.Tensor, b: torch.Tensor, rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier shell correlation between two 3D images with batching support.

    Args:
        a: Input tensor of shape (..., d, h, w)
        b: Input tensor of shape (..., d, h, w)
        rfft_mask: Optional mask for rfft, shape should match rfft output

    Returns
    -------
        Correlation values of shape (..., min(d, h, w) // 2 + 1)
    """
    return _fourier_correlation(a, b, rfft_mask, ndim=3)


def _fourier_correlation(
    a: torch.Tensor,
    b: torch.Tensor,
    rfft_mask: torch.Tensor | None = None,
    ndim: int = 3,
) -> torch.Tensor:
    """Core fourier correlation implementation with batching support.

    Args:
        a: Input tensor with shape (..., spatial_dims)
        b: Input tensor with shape (..., spatial_dims), same shape as a
        rfft_mask: Optional mask for rfft indices, shape should match rfft output
        ndim: Number of spatial dimensions (2 or 3)

    Returns
    -------
        Fourier correlation values with shape (..., n_shells)
    """
    if a.ndim < ndim:
        raise ValueError(f"Input tensors must have at least {ndim} dimensions.")
    elif a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")

    # Extract spatial dimensions and batch dimensions
    spatial_dims = a.shape[-ndim:]
    batch_dims = a.shape[:-ndim]

    # Validate spatial dimensions are equal (for proper shell/ring correlation)
    if len(set(spatial_dims)) != 1:
        raise ValueError(
            "All spatial dimensions must be equal for proper shell/ring correlation."
        )

    image_shape = spatial_dims
    dft_shape = rfft_shape(image_shape)

    if rfft_mask is not None and rfft_mask.shape != dft_shape:
        raise ValueError("rfft_mask must have same shape as rfft output.")

    # Vectorized FFT computation over spatial dimensions only
    spatial_dims_list = list(range(-ndim, 0))  # e.g., [-3, -2, -1] for ndim=3
    a_fft = torch.fft.rfftn(a, dim=spatial_dims_list)
    b_fft = torch.fft.rfftn(b, dim=spatial_dims_list)

    # Compute frequency grid once (same for all batch items)
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
        norm=True,
        device=a.device,
    )

    # Apply mask and flatten spatial dimensions
    if rfft_mask is not None:
        # Apply mask to FFTs and frequency grid
        a_fft = a_fft[..., rfft_mask]  # (..., n_masked_freqs)
        b_fft = b_fft[..., rfft_mask]
        frequencies = frequency_grid[rfft_mask]
    else:
        # Flatten spatial dimensions while preserving batch dimensions
        original_shape = a_fft.shape
        batch_shape = original_shape[:-ndim]
        spatial_size = torch.prod(torch.tensor(original_shape[-ndim:]))
        a_fft = a_fft.view(*batch_shape, spatial_size)
        b_fft = b_fft.view(*batch_shape, spatial_size)
        frequencies = torch.flatten(frequency_grid)

    # Compute shell indices once (same for all batch items)
    bin_centers = torch.fft.rfftfreq(image_shape[0], device=a.device)
    df = 1 / image_shape[0]
    bin_centers = torch.cat([bin_centers, torch.as_tensor([0.5 + df], device=a.device)])
    bin_centers = bin_centers.unfold(dimension=0, size=2, step=1)  # (n_shells, 2)
    split_points = einops.reduce(
        bin_centers, "shells high_low -> shells", reduction="mean"
    )

    # Find indices of all components in each shell
    sorted_frequencies, sort_idx = torch.sort(frequencies, descending=False)
    split_idx = torch.searchsorted(sorted_frequencies, split_points)
    shell_idx = torch.tensor_split(sort_idx, split_idx.cpu())[:-1]

    # Vectorized computation of normalized cross correlation for each shell
    correlation_results = []

    # Handle DC component (shell 0) - always 1.0
    correlation_results.append(torch.ones(batch_dims, device=a.device))

    # Process each shell (vectorized over batch dimension)
    for idx in shell_idx[1:]:
        if len(idx) == 0:
            # Empty shell
            correlation_results.append(torch.zeros(batch_dims, device=a.device))
        else:
            # Get FFT values for this shell for all batch items
            a_shell = a_fft[..., idx]  # (..., n_freqs_in_shell)
            b_shell = b_fft[..., idx]  # (..., n_freqs_in_shell)

            # Vectorized normalized cross correlation
            correlation = _vectorized_normalized_cc_complex(a_shell, b_shell)
            correlation_results.append(correlation)

    # Stack results along last dimension
    correlation_tensor = torch.stack(correlation_results, dim=-1)  # (..., n_shells)

    return torch.real(correlation_tensor)


def rfft_shape(image_shape: Sequence[int]) -> Tuple[int, ...]:
    """Calculate the shape of an rfft on an input image of a given shape.

    Args:
        image_shape: Shape of the input image

    Returns
    -------
        Shape of the rfft output, with last dimension adjusted for real FFT
    """
    rfft_shape = list(image_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def _vectorized_normalized_cc_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized normalized cross correlation for batched complex tensors.

    Args:
        a: Complex tensor of shape (batch_size, n_freqs_in_shell)
        b: Complex tensor of shape (batch_size, n_freqs_in_shell)

    Returns
    -------
        Normalized correlation for each batch item, shape (batch_size,)
    """
    # Compute dot product along frequency dimension for each batch item
    correlation = torch.sum(a * torch.conj(b), dim=-1)  # (batch_size,)

    # Compute norms along frequency dimension for each batch item
    norm_a = torch.linalg.norm(a, dim=-1)  # (batch_size,)
    norm_b = torch.linalg.norm(b, dim=-1)  # (batch_size,)

    # Normalize, handling potential division by zero
    denominator = norm_a * norm_b
    correlation = torch.where(
        denominator > 0, correlation / denominator, torch.zeros_like(correlation)
    )

    return correlation


def fsc(
    a: torch.Tensor, b: torch.Tensor, rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring/shell correlation between two square/cubic images.

    .. deprecated::
        Use `fourier_ring_correlation` for 2D or `fourier_shell_correlation` for 3D.

    Args:
        a: Input tensor (2D or 3D)
        b: Input tensor (2D or 3D), same shape as a
        rfft_mask: Optional mask for rfft indices

    Returns
    -------
        Correlation values of shape (n_shells,)
    """
    if a.ndim == 2:
        return fourier_ring_correlation(a, b, rfft_mask)
    elif a.ndim == 3:
        return fourier_shell_correlation(a, b, rfft_mask)
    else:
        raise ValueError("images must be 2D or 3D.")
