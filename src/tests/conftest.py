"""Pytest configuration and shared fixtures."""
import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


@pytest.fixture
def device():
    """Return CPU device for testing (no GPU overhead)."""
    return torch.device("cpu")


@pytest.fixture
def dummy_input():
    """Create dummy input tensor for testing."""
    batch_size = 2
    height, width, channels = 4, 4, 3
    return torch.randn(batch_size, height, width, channels)


@pytest.fixture
def quantum_dims():
    """Standard quantum dimensions for testing."""
    return (batch_size := 1, 4, 4, 3)
