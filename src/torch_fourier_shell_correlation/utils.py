"""Utility functions for fourier shell correlation computation."""

from typing import Sequence

import einops
import torch


def _prepare_fft_data(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    frequency_grid: torch.Tensor,
    rfft_mask: torch.Tensor | None,
    ndim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare FFT data by applying mask and flattening spatial dimensions.

    Args:
        a_fft: FFT of first tensor
        b_fft: FFT of second tensor
        frequency_grid: Frequency grid
        rfft_mask: Optional mask for rfft indices
        ndim: Number of spatial dimensions

    Returns
    -------
        Tuple of (a_fft_flat, b_fft_flat, frequencies)
    """
    if rfft_mask is not None:
        # Apply mask to FFTs and frequency grid
        a_fft_flat = a_fft[..., rfft_mask]  # (..., n_masked_freqs)
        b_fft_flat = b_fft[..., rfft_mask]
        frequencies = frequency_grid[rfft_mask]
    else:
        # Flatten spatial dimensions while preserving batch dimensions
        original_shape = a_fft.shape
        batch_shape = original_shape[:-ndim]
        spatial_size = torch.prod(torch.tensor(original_shape[-ndim:]))
        a_fft_flat = a_fft.view(*batch_shape, spatial_size)
        b_fft_flat = b_fft.view(*batch_shape, spatial_size)
        frequencies = torch.flatten(frequency_grid)

    return a_fft_flat, b_fft_flat, frequencies


def _compute_frequency_bins(image_size: int, device: torch.device) -> torch.Tensor:
    """Compute frequency bin centers for shell/ring correlation.

    Args:
        image_size: Size of the image along one dimension
        device: Device to create tensors on

    Returns
    -------
        Frequency bin centers with shape (n_bins,)
    """
    bin_centers = torch.fft.rfftfreq(image_size, device=device)
    df = 1 / image_size

    # Setup to split data at midpoint between frequency bin centers
    bin_centers = torch.cat([bin_centers, torch.as_tensor([0.5 + df], device=device)])
    bin_centers = bin_centers.unfold(dimension=0, size=2, step=1)  # (n_shells, 2)
    split_points = einops.reduce(
        bin_centers, "shells high_low -> shells", reduction="mean"
    )

    return split_points


def _compute_shell_indices(
    frequencies: torch.Tensor, bin_centers: torch.Tensor
) -> list[torch.Tensor]:
    """Compute indices for each frequency shell.

    Args:
        frequencies: Flattened frequency magnitudes
        bin_centers: Frequency bin centers

    Returns
    -------
        List of index tensors for each shell
    """
    # Find indices of all components in each shell
    sorted_frequencies, sort_idx = torch.sort(frequencies, descending=False)
    split_idx = torch.searchsorted(sorted_frequencies, bin_centers)
    shell_indices = list(torch.tensor_split(sort_idx, split_idx.cpu())[:-1])

    return shell_indices


def _compute_shell_correlations(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    shell_indices: list[torch.Tensor],
    batch_dims: Sequence[int],
) -> torch.Tensor:
    """Compute normalized cross correlation for each shell.

    Args:
        a_fft: Flattened FFT data for first tensor
        b_fft: Flattened FFT data for second tensor
        shell_indices: List of indices for each shell
        batch_dims: Batch dimensions shape

    Returns
    -------
        Correlation values for each shell with shape (*batch_dims, n_shells)
    """
    correlation_results = []

    # Handle DC component (shell 0) - always 1.0
    correlation_results.append(torch.ones(batch_dims, device=a_fft.device))

    # Process each shell (vectorized over batch dimension)
    for idx in shell_indices[1:]:
        if len(idx) == 0:
            # Empty shell
            correlation_results.append(torch.zeros(batch_dims, device=a_fft.device))
        else:
            # Get FFT values for this shell for all batch items
            a_shell = a_fft[..., idx]  # (..., n_freqs_in_shell)
            b_shell = b_fft[..., idx]  # (..., n_freqs_in_shell)

            # Vectorized normalized cross correlation
            correlation = _normalized_cc_complex(a_shell, b_shell)
            correlation_results.append(correlation)

    # Stack results along last dimension
    return torch.stack(correlation_results, dim=-1)  # (..., n_shells)


def _normalized_cc_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
