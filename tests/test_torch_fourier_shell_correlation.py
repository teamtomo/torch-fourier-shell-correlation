import torch

from torch_fourier_shell_correlation import (
    fourier_ring_correlation,
    fourier_shell_correlation,
)


def test_fourier_ring_correlation_2d():
    """Test 2D fourier ring correlation without batching."""
    a = torch.zeros((10, 10))
    a[4:6, 4:6] = 1

    b = a
    result = fourier_ring_correlation(a, b)
    assert torch.allclose(result, torch.ones(6))

    b = torch.rand((10, 10))
    result = fourier_ring_correlation(a, b)
    assert not torch.allclose(result, torch.ones(6))


def test_fourier_shell_correlation_3d():
    """Test 3D fourier shell correlation without batching."""
    a = torch.zeros((10, 10, 10))
    a[4:6, 4:6, 4:6] = 1

    b = a
    result = fourier_shell_correlation(a, b)
    assert torch.allclose(result, torch.ones(6))

    b = torch.rand((10, 10, 10))
    result = fourier_shell_correlation(a, b)
    assert not torch.allclose(result, torch.ones(6))


def test_fourier_ring_correlation_batched():
    """Test 2D fourier ring correlation with batching."""
    # Single batch
    a = torch.zeros((2, 10, 10))
    a[:, 4:6, 4:6] = 1
    b = a.clone()

    result = fourier_ring_correlation(a, b)
    assert result.shape == (2, 6)
    assert torch.allclose(result, torch.ones(2, 6))

    # Multiple batch dimensions
    a = torch.zeros((3, 2, 10, 10))
    a[:, :, 4:6, 4:6] = 1
    b = a.clone()

    result = fourier_ring_correlation(a, b)
    assert result.shape == (3, 2, 6)
    assert torch.allclose(result, torch.ones(3, 2, 6))


def test_fourier_shell_correlation_batched():
    """Test 3D fourier shell correlation with batching."""
    # Single batch
    a = torch.zeros((2, 10, 10, 10))
    a[:, 4:6, 4:6, 4:6] = 1
    b = a.clone()

    result = fourier_shell_correlation(a, b)
    assert result.shape == (2, 6)
    assert torch.allclose(result, torch.ones(2, 6))

    # Multiple batch dimensions
    a = torch.zeros((3, 2, 10, 10, 10))
    a[:, :, 4:6, 4:6, 4:6] = 1
    b = a.clone()

    result = fourier_shell_correlation(a, b)
    assert result.shape == (3, 2, 6)
    assert torch.allclose(result, torch.ones(3, 2, 6))


def test_batched_vs_individual_processing():
    """Test that batched processing gets the same results as individual processing."""
    torch.manual_seed(42)  # For reproducible results

    # Test 2D case
    batch_size = 3
    a_2d_batch = torch.rand((batch_size, 16, 16))
    b_2d_batch = torch.rand((batch_size, 16, 16))

    # Process as batch
    result_batch = fourier_ring_correlation(a_2d_batch, b_2d_batch)

    # Process individually
    individual_results = []
    for i in range(batch_size):
        result_individual = fourier_ring_correlation(a_2d_batch[i], b_2d_batch[i])
        individual_results.append(result_individual)
    result_individual_stacked = torch.stack(individual_results)

    # Should be identical
    assert torch.allclose(result_batch, result_individual_stacked, atol=1e-6)

    # Test 3D case
    a_3d_batch = torch.rand((batch_size, 12, 12, 12))
    b_3d_batch = torch.rand((batch_size, 12, 12, 12))

    # Process as batch
    result_batch_3d = fourier_shell_correlation(a_3d_batch, b_3d_batch)

    # Process individually
    individual_results_3d = []
    for i in range(batch_size):
        result_individual_3d = fourier_shell_correlation(a_3d_batch[i], b_3d_batch[i])
        individual_results_3d.append(result_individual_3d)
    result_individual_stacked_3d = torch.stack(individual_results_3d)

    # Should be identical
    assert torch.allclose(result_batch_3d, result_individual_stacked_3d, atol=1e-6)


def test_multidimensional_batched_vs_individual():
    """Test multi-dimensional batching gets same results as individual processing."""
    torch.manual_seed(42)

    # Test 2D case with multiple batch dimensions
    batch_shape = (2, 3)
    a_2d_multi = torch.rand((*batch_shape, 16, 16))
    b_2d_multi = torch.rand((*batch_shape, 16, 16))

    # Process as multi-batch
    result_multi_batch = fourier_ring_correlation(a_2d_multi, b_2d_multi)

    # Process individually and reconstruct
    individual_results = []
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            result_individual = fourier_ring_correlation(
                a_2d_multi[i, j], b_2d_multi[i, j]
            )
            individual_results.append(result_individual)

    # Reshape to match multi-batch result
    result_individual_reshaped = torch.stack(individual_results).view(*batch_shape, -1)

    # Should be identical
    assert torch.allclose(result_multi_batch, result_individual_reshaped, atol=1e-6)


def test_edge_cases():
    """Test edge cases like identical images, zeros, and ones."""
    # Test batched edge cases
    a_batch = torch.stack([torch.ones((12, 12, 12)), torch.zeros((12, 12, 12))])
    b_batch = a_batch.clone()

    result_batch = fourier_shell_correlation(a_batch, b_batch)

    # Process individually for comparison
    result_individual_0 = fourier_shell_correlation(a_batch[0], b_batch[0])
    result_individual_1 = fourier_shell_correlation(a_batch[1], b_batch[1])

    assert torch.allclose(result_batch[0], result_individual_0)
    assert torch.allclose(result_batch[1], result_individual_1)
