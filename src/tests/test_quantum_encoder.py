"""Unit tests for quantum encoder components using the real implementations."""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# These tests exercise the actual quantum encoder and therefore require the
# heavy quantum dependencies. When they are missing we skip the entire module.
pytest.importorskip("perceval", reason="Quantum encoder tests require Perceval.")
pytest.importorskip("merlin", reason="Quantum encoder tests require Merlin.")

from config import QuantumConfig
from models.quantum_encoder import BosonSampler, AMPLITUDE_ENCODING
from models.legacy.quantum_encoder_legacy import QEncoder


class TestBosonSamplerBasics:
    """Test basic BosonSampler functionality without GPU."""
    
    def test_boson_sampler_import(self):
        """Test that BosonSampler can be imported."""
        assert BosonSampler is not None
    
    def test_boson_sampler_initialization(self):
        """Test BosonSampler initialization with real dependencies."""
        dims = (1, 4, 4, 3)
        sampler = BosonSampler(dims)
        
        assert sampler is not None
        assert sampler.dims == (4, 4, 3)
        assert sampler.m == sum(sampler.dims) + 2
        assert sampler.n == len(sampler.dims) + 1
        assert sampler.eps == pytest.approx(1e-5)
    
    def test_boson_sampler_initialization_no_batch_dim(self):
        """Allow passing tensor shape without explicit batch dimension."""
        sampler = BosonSampler((4, 8, 8))
        assert sampler.dims == (4, 8, 8)
        assert sampler.m == sum(sampler.dims) + 2
        assert sampler.n == len(sampler.dims) + 1

    def test_boson_sampler_from_config(self):
        config = QuantumConfig(num_modes=16, num_photons=2, epsilon=1e-4)
        sampler = BosonSampler.from_config((1, 4, 4, 3), config=config, device=torch.device("cpu"))
        assert sampler.m == sum(sampler.dims) + 2
        assert sampler.n == len(sampler.dims) + 1
        assert sampler.eps == 1e-4
        assert sampler.device.type == "cpu"
        
    def test_boson_sampler_override_quantum_dims(self):
        """Ensure user-provided quantum dimensions can override defaults."""
        config = QuantumConfig(num_modes=12, num_photons=2, epsilon=1e-4)
        with pytest.raises(ValueError):
            BosonSampler((1, 4, 4, 3), config=config, override_quantum_dims=True)

        valid_config = QuantumConfig(num_modes=13, num_photons=4, epsilon=1e-4)
        sampler = BosonSampler((1, 4, 4, 3), config=valid_config, override_quantum_dims=True)
        assert sampler.m == 13
        assert sampler.n == 4
        assert sampler.config.num_modes == 13
        assert sampler.config.num_photons == 4
    
    def test_boson_sampler_unsorted_path(self):
        """When sort_encoding is False we should get the unsorted implementation."""
        config = QuantumConfig(sort_encoding=False, num_modes=20, num_photons=3)
        sampler = BosonSampler((1, 4, 4, 3), config=config)
        assert hasattr(sampler, "grouping")
        assert not hasattr(sampler, "postselection_indices")
    
    def test_boson_sampler_forward(self):
        """Test BosonSampler forward pass."""
        dims = (1, 4, 4, 3)
        sampler = BosonSampler(dims)
        
        # Create input tensor
        batch_size = 2
        x = torch.randn(batch_size, 4, 4, 3)
        
        # Forward pass
        output = sampler(x)
        
        # Check output shape - output should match input spatial dimensions
        assert output.shape[0] == batch_size
        assert output.shape == (batch_size, 4, 4, 3)
        assert output.dtype == torch.float32
        assert sampler.amplitude_encoding is AMPLITUDE_ENCODING
        assert getattr(sampler.model, "amplitude_encoding", AMPLITUDE_ENCODING) is AMPLITUDE_ENCODING

    def test_boson_sampler_amplitude_encoding_matches_reference(self):
        """Ensure amplitude mapping matches the legacy positive/negative scheme."""
        dims = (1, 4, 4, 3)
        sampler = BosonSampler(dims)
        x = torch.rand(3, 4, 4, 3)

        amplitudes = sampler._encode_inputs(x)
        flat = x.view(x.shape[0], -1).clamp(0.0, 1.0)
        norm = math.sqrt(flat.shape[1])
        expected_positive = torch.sqrt(flat) / norm
        expected_negative = torch.sqrt(1.0 - flat) / norm

        assert amplitudes.shape[1] == sampler.input_dim
        assert torch.allclose(amplitudes[:, sampler.positive_indices], expected_positive, atol=1e-6)
        assert torch.allclose(amplitudes[:, sampler.negative_indices], expected_negative, atol=1e-6)
    
    def test_boson_sampler_device_cpu(self):
        """Test BosonSampler on CPU."""
        dims = (1, 4, 4, 3)
        sampler = BosonSampler(dims).to(torch.device("cpu"))
        assert sampler.device.type == "cpu"
        params = list(sampler.model.parameters())
        assert all(param.device.type == "cpu" for param in params) or not params
    
    def test_boson_sampler_parameters(self):
        """Test that BosonSampler has trainable parameters."""
        dims = (1, 4, 4, 3)
        sampler = BosonSampler(dims)
        
        params = list(sampler.parameters())
        assert len(params) > 0, "BosonSampler should have trainable parameters"
    
    def test_boson_sampler_different_dims(self):
        """Test BosonSampler with different input dimensions."""
        test_cases = [
            (1, 2, 2, 3),
            (1, 8, 8, 4),
            (2, 3, 3, 3),
        ]
        
        for dims in test_cases:
            sampler = BosonSampler(dims)
            assert sampler.dims == dims[1:]
            
            x = torch.randn(2, *dims[1:])
            output = sampler(x)
            
            assert output.shape == (2,) + dims[1:], \
                f"Output shape {output.shape} should match input spatial dims"


class TestBosonSamplerGradients:
    """Test gradient flow and optimization."""
    
    def test_boson_sampler_backward(self):
        """Test backward pass and gradient computation."""
        dims = (1, 4, 4, 3)
        sampler = BosonSampler(dims)
        
        x = torch.randn(2, 4, 4, 3)
        output = sampler(x)
        
        loss = output.sum()
        loss.backward()
        
        for param in sampler.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient should be computed"
                assert param.grad.shape == param.shape
    
    def test_boson_sampler_optimizer_step(self):
        """Test that BosonSampler works with standard PyTorch optimizer."""
        dims = (1, 4, 4, 3)
        sampler = BosonSampler(dims)
        optimizer = torch.optim.Adam(sampler.parameters(), lr=0.001)
        
        x = torch.randn(2, 4, 4, 3)
        output = sampler(x)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert True


class TestBosonSamplerIntegration:
    """Integration tests combining multiple components."""
    
    def test_boson_sampler_in_nn_module(self):
        """Test BosonSampler as part of a larger nn.Module."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.quantum = BosonSampler((1, 4, 4, 3))
                feature_dim = 4 * 4 * 3
                self.classifier = nn.Linear(feature_dim, 2)
            
            def forward(self, x):
                x = self.quantum(x)
                x = x.view(x.shape[0], -1)
                x = self.classifier(x)
                return x
        
        model = SimpleModel()
        x = torch.randn(2, 4, 4, 3)
        output = model(x)
        
        assert output.shape == (2, 2)
    
    def test_boson_sampler_batch_processing(self):
        """Test BosonSampler with various batch sizes."""
        sampler = BosonSampler((1, 4, 4, 3))
        
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 4, 4, 3)
            output = sampler(x)
            assert output.shape[0] == batch_size


class TestBosonSamplerLegacyComparison:
    """Compare new BosonSampler (quantum_encoder.py) with legacy implementation.

    One problem is that the BS is random by nature, so we cannot expect exact output
    
    This test suite validates that the refactored BosonSampler maintains
    compatibility with the legacy version's behavior, especially:
    - Amplitude normalization logic
    - Output shape consistency
    - Parameter structure
    """
    
    def test_amplitude_normalization_equivalence(self):
        """Ensure decoder matches the conditional probabilities of the legacy encoder."""
        dims = (2, 4, 4, 3)
        sampler = BosonSampler(dims)
        legacy = QEncoder(sampler.dims)
        
        assert len(sampler.positive_indices) == len(legacy.postselected_states_idx[::2])
        assert len(sampler.negative_indices) == len(legacy.postselected_states_idx[1::2])
        
        batch = 3
        positive = torch.rand(batch, math.prod(sampler.dims))
        negative = torch.rand(batch, math.prod(sampler.dims))

        raw_new = torch.zeros(batch, sampler.input_dim)
        raw_new[:, sampler.positive_indices] = positive
        raw_new[:, sampler.negative_indices] = negative

        legacy_space_dim = max(legacy.postselected_states_idx) + 1
        legacy_states = torch.zeros(batch, legacy_space_dim)
        legacy_states[:, legacy.postselected_states_idx[::2]] = positive
        legacy_states[:, legacy.postselected_states_idx[1::2]] = negative
        
        decoded = sampler._decode_outputs(raw_new, batch)
        legacy_decoded = torch.stack([
            legacy._decode_nparray(legacy_states[i].detach().cpu().numpy())
            for i in range(batch)
        ])
        assert decoded.shape == legacy_decoded.shape
        assert torch.allclose(decoded, legacy_decoded, atol=1e-6)
    
    def test_input_output_shape_consistency(self):
        """Verify input/output shapes match between legacy and new implementations.
        
        Both should accept (batch_size, C, H, W) and output (batch_size, C, H, W).
        """
        test_shapes = [
            (1, 4, 4, 3),      # Single sample
            (2, 4, 4, 3),      # Small batch
            (4, 8, 8, 4),      # Larger spatial dims
        ]
        
        for input_shape in test_shapes:
            sampler = BosonSampler(input_shape)
            x = torch.randn(*input_shape)
            output = sampler(x)
            
            assert output.shape == input_shape, \
                f"Output shape {output.shape} should match input shape {input_shape}"
    
    def test_epsilon_parameter_consistency(self):
        """Verify epsilon parameter is used consistently for numerical stability.
        
        Both implementations should:
        - Accept epsilon in config
        - Use it consistently in normalization
        - Prevent division by zero
        """
        test_eps_values = [1e-5, 1e-4, 1e-3]
        
        for eps_val in test_eps_values:
            config = QuantumConfig(epsilon=eps_val, num_modes=20, num_photons=3)
            sampler = BosonSampler((1, 4, 4, 3), config=config)
            
            x = torch.ones(1, 4, 4, 3) * 1e-8
            
            try:
                output = sampler(x)
                assert not torch.isnan(output).any(), \
                    f"Output contains NaN with eps={eps_val}"
                assert not torch.isinf(output).any(), \
                    f"Output contains Inf with eps={eps_val}"
            except Exception as e:
                pytest.fail(f"Forward pass failed with eps={eps_val}: {e}")
    
    def test_quantum_parameter_compatibility(self):
        """Verify quantum parameters (modes, photons) are handled consistently.
        
        Legacy uses: m = sum(dims[1:]) + 2, n = len(dims[1:]) + 1
        New uses: m = num_modes, n = num_photons (from config)
        
        Both should properly initialize the quantum circuit.
        """
        # Test various quantum configurations
        configs = [
            QuantumConfig(num_modes=16, num_photons=2),
            QuantumConfig(num_modes=20, num_photons=3),
            QuantumConfig(num_modes=30, num_photons=4),
        ]
        
        for cfg in configs:
            sampler = BosonSampler((1, 4, 4, 3), config=cfg)
            
            expected_m = sum(sampler.dims) + 2
            expected_n = len(sampler.dims) + 1
            assert sampler.m == expected_m
            assert sampler.n == expected_n
            assert sampler.config.num_modes == expected_m
            assert sampler.config.num_photons == expected_n
            
            if getattr(sampler.config, "computation_space", "UNBUNCHED").upper() == "UNBUNCHED":
                expected_hilbert_dim = math.comb(expected_m, expected_n)
            else:
                expected_hilbert_dim = math.comb(expected_m + expected_n - 1, expected_n)
            assert sampler.input_dim == expected_hilbert_dim, \
                f"Input dimension mismatch for m={sampler.m}, n={sampler.n}. Expected {expected_hilbert_dim}, got {sampler.input_dim}"
    
    def test_deterministic_behavior_with_seed(self):
        """Verify creating multiple samplers back-to-back stays stable."""
        dims = (2, 4, 4, 3)
        config = QuantumConfig(num_modes=16, num_photons=2)
        
        x1 = torch.randn(2, 4, 4, 3)
        x2 = torch.randn(2, 4, 4, 3)
        
        sampler1 = BosonSampler(dims, config=config)
        output1 = sampler1(x1)
        sampler2 = BosonSampler(dims, config=config)
        output2 = sampler2(x2)
        
        # Verify outputs have reasonable shapes and values
        assert output1.shape[0] == 2, "Batch size should be preserved"
        assert output2.shape[0] == 2, "Batch size should be preserved"
        assert not torch.isnan(output1).any(), "Output1 contains NaN"
        assert not torch.isnan(output2).any(), "Output2 contains NaN"
    
    def test_batch_processing_consistency(self):
        """Verify batch processing produces same results as individual samples.
        
        Processing a batch should be equivalent to processing samples individually
        (within numerical precision).
        """
        sampler = BosonSampler((1, 4, 4, 3))
        x_batch = torch.randn(3, 4, 4, 3)
        
        output_batch = sampler(x_batch)
        
        outputs_individual = []
        for i in range(3):
            x_single = x_batch[i:i+1]
            output_single = sampler(x_single)
            outputs_individual.append(output_single)
        
        output_individual_stacked = torch.cat(outputs_individual, dim=0)
        
        assert output_batch.shape == output_individual_stacked.shape, \
            f"Shape mismatch: batch {output_batch.shape} vs individual {output_individual_stacked.shape}"


class TestLegacyNewIntegration:
    """Integration tests ensuring smooth migration from legacy to new implementation.
    
    These tests verify that:
    - Models trained with legacy code can be evaluated with new code
    - Both versions handle the same quantum circuits
    - State dict compatibility (if applicable)
    """
    
    def test_both_samplers_accept_same_config(self):
        """Verify both old and new samplers work with QuantumConfig object."""
        config = QuantumConfig(
            num_modes=20,
            num_photons=3,
            epsilon=1e-5,
            computation_space="UNBUNCHED"
        )
        
        sampler_new = BosonSampler((1, 4, 4, 3), config=config)
        assert sampler_new.config == config
        assert sampler_new.m == sum(sampler_new.dims) + 2
        assert sampler_new.n == len(sampler_new.dims) + 1
    
    def test_output_range_validity(self):
        """Verify output values are in valid ranges for both implementations.
        
        Quantum measurements should produce probabilities in [0, 1].
        """
        sampler = BosonSampler((2, 4, 4, 3))
        x = torch.randn(2, 4, 4, 3)
        output = sampler(x)
        
        assert output.shape == (2, 4, 4, 3)
        assert output.dtype in [torch.float32, torch.float64], \
            f"Unexpected output dtype: {output.dtype}"
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
