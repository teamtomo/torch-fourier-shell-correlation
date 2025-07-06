"""Fourier shell correlation between two 2D or 3D images."""

from typing import Sequence

import torch
from torch_grid_utils import fftfreq_grid

from .utils import (
    _compute_frequency_bins_weighted,
    _compute_shell_correlations_weighted,
    _prepare_fft_data,
)


def fourier_ring_correlation(
    a: torch.Tensor, b: torch.Tensor, rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring correlation between two 2D images with batching support.

    Supports both square and rectangular images using weighted interpolation.

    Args:
        a: Input tensor of shape (..., h, w)
        b: Input tensor of shape (..., h, w)
        rfft_mask: Optional mask for rfft, shape should match rfft output

    Returns
    -------
        Correlation values of shape (..., min(h, w) // 2 + 1)
    """
    # Input validation
    if a.ndim < 2:
        raise ValueError("Input tensors must have at least 2 dimensions.")
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")

    # Compute FFT
    spatial_dims_list = [-2, -1]  # Last 2 dimensions
    a_fft = torch.fft.rfftn(a, dim=spatial_dims_list)
    b_fft = torch.fft.rfftn(b, dim=spatial_dims_list)

    # Get spatial dimensions
    spatial_dims = a.shape[-2:]

    return fourier_correlation(a_fft, b_fft, spatial_dims, rfft_mask)


def fourier_shell_correlation(
    a: torch.Tensor, b: torch.Tensor, rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier shell correlation between two 3D images with batching support.

    Supports both cubic and rectangular volumes using weighted interpolation.

    Args:
        a: Input tensor of shape (..., d, h, w)
        b: Input tensor of shape (..., d, h, w)
        rfft_mask: Optional mask for rfft, shape should match rfft output

    Returns
    -------
        Correlation values of shape (..., min(d, h, w) // 2 + 1)
    """
    # Input validation
    if a.ndim < 3:
        raise ValueError("Input tensors must have at least 3 dimensions.")
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")

    # Compute FFT
    spatial_dims_list = [-3, -2, -1]  # Last 3 dimensions
    a_fft = torch.fft.rfftn(a, dim=spatial_dims_list)
    b_fft = torch.fft.rfftn(b, dim=spatial_dims_list)

    # Get spatial dimensions
    spatial_dims = a.shape[-3:]

    return fourier_correlation(a_fft, b_fft, spatial_dims, rfft_mask)


def fourier_correlation(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    spatial_dims: Sequence[int],
    rfft_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute fourier correlation from FFT data supporting rectangular shapes.

    Args:
        a_fft: FFT of first tensor with shape (..., *fft_shape)
        b_fft: FFT of second tensor with shape (..., *fft_shape)
        spatial_dims: Shape of the spatial dimensions before FFT
        rfft_mask: Optional mask for rfft indices

    Returns
    -------
        Fourier correlation values with shape (..., n_shells)
    """
    # Input validation
    if a_fft.shape != b_fft.shape:
        raise ValueError("FFT tensors must have the same shape.")

    # Validate rfft_mask
    fft_shape = a_fft.shape[-len(spatial_dims) :]
    if rfft_mask is not None and rfft_mask.shape != fft_shape:
        raise ValueError("rfft_mask must have same shape as rfft output.")

    # Compute frequency grid and prepare FFT data
    frequency_grid = fftfreq_grid(
        image_shape=spatial_dims,
        rfft=True,
        fftshift=False,
        norm=True,
        device=a_fft.device,
    )

    # Apply mask and flatten spatial dimensions
    a_fft_flat, b_fft_flat, frequencies = _prepare_fft_data(
        a_fft, b_fft, frequency_grid, rfft_mask, len(spatial_dims)
    )

    # Compute frequency bins using weighted approach
    bin_centers = _compute_frequency_bins_weighted(spatial_dims, a_fft.device)

    # Compute correlations using weighted interpolation
    correlations = _compute_shell_correlations_weighted(
        a_fft_flat,
        b_fft_flat,
        frequencies,
        bin_centers,
        a_fft.shape[: -len(spatial_dims)],
    )

    return torch.real(correlations)


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
