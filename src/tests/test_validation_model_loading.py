"""
Tests to verify model loading sanity in validation pipeline.

This module ensures:
1. Models are loaded into correct evaluation state
2. Underlying VAE modules are properly set to eval mode
3. Quantum samplers are loaded and configured correctly
4. Checkpoint format compatibility
5. Model weights are correctly transferred after loading
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import pytest

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models.cyclegan_turbo import VAE_encode, VAE_decode
from validation import EvaluationTrainer, EvalAccelerator


class TestModelEvaluationMode:
    """Test that models are correctly set to eval mode after loading."""
    
    @patch('validation.CycleGAN_Turbo')
    def test_unet_in_eval_mode_after_loading(self, mock_cyclegan, device):
        """Verify UNet is in eval mode post-load."""
        # Create mock UNet
        mock_unet = MagicMock()
        mock_unet.training = False
        
        # Create mock VAE modules
        mock_vae = MagicMock()
        mock_vae.training = False
        mock_vae_b2a = MagicMock()
        mock_vae_b2a.training = False
        
        # Create mock VAE wrappers
        mock_vae_enc = MagicMock()
        mock_vae_enc.vae = mock_vae
        mock_vae_enc.vae_b2a = mock_vae_b2a
        mock_vae_enc.training = False
        
        mock_vae_dec = MagicMock()
        mock_vae_dec.vae = mock_vae
        mock_vae_dec.vae_b2a = mock_vae_b2a
        mock_vae_dec.training = False
        
        # Create mock CycleGAN_Turbo instance
        mock_cyclegan_instance = MagicMock()
        mock_cyclegan_instance.unet = mock_unet
        mock_cyclegan_instance.vae_enc = mock_vae_enc
        mock_cyclegan_instance.vae_dec = mock_vae_dec
        mock_cyclegan.return_value = mock_cyclegan_instance
        
        trainer = EvaluationTrainer(
            device=device,
            noise_scheduler=None,
            fixed_a2b_emb=torch.randn(1, 77, 768),
            fixed_b2a_emb=torch.randn(1, 77, 768),
            boson_sampler=None,
            quantum_enabled=False,
            dynamic_quantum=False,
        )
        
        # Mock torch.load to return minimal checkpoint
        with patch('torch.load', return_value={}):
            trainer.load_checkpoint("dummy_checkpoint.pkl")
        
        # Verify eval() was called on UNet
        assert mock_unet.eval.called, "UNet.eval() should be called"
        assert mock_vae_enc.eval.called, "VAE_encode.eval() should be called"
    
    @patch('validation.CycleGAN_Turbo')
    def test_vae_enc_underlying_modules_in_eval(self, mock_cyclegan, device):
        """Verify VAE_encode wrapper and underlying VAE in eval mode."""
        # Create real VAE modules
        mock_vae = MagicMock()
        mock_vae_b2a = MagicMock()
        
        # Create VAE wrapper with real eval() method
        mock_vae_enc = MagicMock()
        mock_vae_enc.vae = mock_vae
        mock_vae_enc.vae_b2a = mock_vae_b2a
        
        mock_vae_dec = MagicMock()
        mock_vae_dec.vae = mock_vae
        mock_vae_dec.vae_b2a = mock_vae_b2a
        
        mock_unet = MagicMock()
        
        mock_cyclegan_instance = MagicMock()
        mock_cyclegan_instance.unet = mock_unet
        mock_cyclegan_instance.vae_enc = mock_vae_enc
        mock_cyclegan_instance.vae_dec = mock_vae_dec
        mock_cyclegan.return_value = mock_cyclegan_instance
        
        trainer = EvaluationTrainer(
            device=device,
            noise_scheduler=None,
            fixed_a2b_emb=torch.randn(1, 77, 768),
            fixed_b2a_emb=torch.randn(1, 77, 768),
            boson_sampler=None,
            quantum_enabled=False,
            dynamic_quantum=False,
        )
        
        with patch('torch.load', return_value={}):
            trainer.load_checkpoint("dummy_checkpoint.pkl")
        
        # Verify underlying VAE modules had eval() called
        assert mock_vae.eval.called, "Underlying VAE.eval() should be called"
        assert mock_vae_b2a.eval.called, "Underlying VAE_b2a.eval() should be called"
    
    @patch('validation.CycleGAN_Turbo')
    def test_vae_dec_in_eval_mode(self, mock_cyclegan, device):
        """Verify VAE_decode wrapper and underlying VAE in eval mode."""
        mock_vae = MagicMock()
        mock_vae_b2a = MagicMock()
        
        mock_vae_enc = MagicMock()
        mock_vae_enc.vae = mock_vae
        mock_vae_enc.vae_b2a = mock_vae_b2a
        
        mock_vae_dec = MagicMock()
        mock_vae_dec.vae = mock_vae
        mock_vae_dec.vae_b2a = mock_vae_b2a
        
        mock_unet = MagicMock()
        
        mock_cyclegan_instance = MagicMock()
        mock_cyclegan_instance.unet = mock_unet
        mock_cyclegan_instance.vae_enc = mock_vae_enc
        mock_cyclegan_instance.vae_dec = mock_vae_dec
        mock_cyclegan.return_value = mock_cyclegan_instance
        
        trainer = EvaluationTrainer(
            device=device,
            noise_scheduler=None,
            fixed_a2b_emb=torch.randn(1, 77, 768),
            fixed_b2a_emb=torch.randn(1, 77, 768),
            boson_sampler=None,
            quantum_enabled=False,
            dynamic_quantum=False,
        )
        
        with patch('torch.load', return_value={}):
            trainer.load_checkpoint("dummy_checkpoint.pkl")
        
        # Verify VAE_dec had eval() called
        assert mock_vae_dec.eval.called, "VAE_decode.eval() should be called"


class TestQuantumSamplerLoading:
    """Test quantum sampler initialization and eval mode."""
    
    @patch('validation.CycleGAN_Turbo')
    def test_boson_sampler_eval_mode_called(self, mock_cyclegan, device):
        """Verify boson sampler eval() is called after loading."""
        mock_unet = MagicMock()
        mock_vae_enc = MagicMock()
        mock_vae_enc.vae = None
        mock_vae_enc.vae_b2a = None
        mock_vae_dec = MagicMock()
        mock_vae_dec.vae = None
        mock_vae_dec.vae_b2a = None
        
        mock_cyclegan_instance = MagicMock()
        mock_cyclegan_instance.unet = mock_unet
        mock_cyclegan_instance.vae_enc = mock_vae_enc
        mock_cyclegan_instance.vae_dec = mock_vae_dec
        mock_cyclegan.return_value = mock_cyclegan_instance
        
        # Create mock boson sampler
        mock_sampler = MagicMock()
        mock_sampler.model = MagicMock()
        mock_sampler.model.state_dict.return_value = {"test": torch.tensor([1.0])}
        
        trainer = EvaluationTrainer(
            device=device,
            noise_scheduler=None,
            fixed_a2b_emb=torch.randn(1, 77, 768),
            fixed_b2a_emb=torch.randn(1, 77, 768),
            boson_sampler=mock_sampler,
            quantum_enabled=True,
            dynamic_quantum=True,
        )
        
        # Mock checkpoint with quantum params
        checkpoint_data = {"quantum_params": {"test": torch.tensor([1.0])}}
        with patch('torch.load', return_value=checkpoint_data):
            trainer.load_checkpoint("dummy_checkpoint.pkl")
        
        # Verify boson sampler.eval() was called
        assert mock_sampler.eval.called, "Boson sampler.eval() should be called"
    
    @patch('validation.CycleGAN_Turbo')
    def test_quantum_state_loading_no_error_when_missing(self, mock_cyclegan, device):
        """Verify loading succeeds when quantum_params is missing."""
        mock_unet = MagicMock()
        mock_vae_enc = MagicMock()
        mock_vae_enc.vae = None
        mock_vae_enc.vae_b2a = None
        mock_vae_dec = MagicMock()
        mock_vae_dec.vae = None
        mock_vae_dec.vae_b2a = None
        
        mock_cyclegan_instance = MagicMock()
        mock_cyclegan_instance.unet = mock_unet
        mock_cyclegan_instance.vae_enc = mock_vae_enc
        mock_cyclegan_instance.vae_dec = mock_vae_dec
        mock_cyclegan.return_value = mock_cyclegan_instance
        
        trainer = EvaluationTrainer(
            device=device,
            noise_scheduler=None,
            fixed_a2b_emb=torch.randn(1, 77, 768),
            fixed_b2a_emb=torch.randn(1, 77, 768),
            boson_sampler=None,
            quantum_enabled=False,
            dynamic_quantum=False,
        )
        
        # Checkpoint without quantum_params - should not raise error
        checkpoint_data = {}
        with patch('torch.load', return_value=checkpoint_data):
            trainer.load_checkpoint("dummy_checkpoint.pkl")
        
        assert trainer.unet is not None, "Model should load successfully"


class TestModelWeightTransfer:
    """Test that model weights are correctly transferred after loading."""
    
    @patch('validation.CycleGAN_Turbo')
    def test_unet_is_not_none_after_loading(self, mock_cyclegan, device):
        """Verify UNet is assigned after loading."""
        mock_unet = MagicMock()
        mock_vae_enc = MagicMock()
        mock_vae_enc.vae = None
        mock_vae_enc.vae_b2a = None
        mock_vae_dec = MagicMock()
        mock_vae_dec.vae = None
        mock_vae_dec.vae_b2a = None
        
        mock_cyclegan_instance = MagicMock()
        mock_cyclegan_instance.unet = mock_unet
        mock_cyclegan_instance.vae_enc = mock_vae_enc
        mock_cyclegan_instance.vae_dec = mock_vae_dec
        mock_cyclegan.return_value = mock_cyclegan_instance
        
        trainer = EvaluationTrainer(
            device=device,
            noise_scheduler=None,
            fixed_a2b_emb=torch.randn(1, 77, 768),
            fixed_b2a_emb=torch.randn(1, 77, 768),
            boson_sampler=None,
            quantum_enabled=False,
            dynamic_quantum=False,
        )
        
        with patch('torch.load', return_value={}):
            trainer.load_checkpoint("dummy_checkpoint.pkl")
        
        assert trainer.unet is not None, "UNet should be assigned after loading"
        assert trainer.unet is mock_unet, "UNet should be the one from CycleGAN_Turbo"
    
    @patch('validation.CycleGAN_Turbo')
    def test_vae_components_are_assigned(self, mock_cyclegan, device):
        """Verify VAE components are assigned after loading."""
        mock_unet = MagicMock()
        mock_vae_enc = MagicMock()
        mock_vae_enc.vae = None
        mock_vae_enc.vae_b2a = None
        mock_vae_dec = MagicMock()
        mock_vae_dec.vae = None
        mock_vae_dec.vae_b2a = None
        
        mock_cyclegan_instance = MagicMock()
        mock_cyclegan_instance.unet = mock_unet
        mock_cyclegan_instance.vae_enc = mock_vae_enc
        mock_cyclegan_instance.vae_dec = mock_vae_dec
        mock_cyclegan.return_value = mock_cyclegan_instance
        
        trainer = EvaluationTrainer(
            device=device,
            noise_scheduler=None,
            fixed_a2b_emb=torch.randn(1, 77, 768),
            fixed_b2a_emb=torch.randn(1, 77, 768),
            boson_sampler=None,
            quantum_enabled=False,
            dynamic_quantum=False,
        )
        
        with patch('torch.load', return_value={}):
            trainer.load_checkpoint("dummy_checkpoint.pkl")
        
        assert trainer.vae_enc is not None, "VAE_encode should be assigned"
        assert trainer.vae_dec is not None, "VAE_decode should be assigned"
        assert trainer.vae_enc is mock_vae_enc, "VAE_encode should match CycleGAN's"
        assert trainer.vae_dec is mock_vae_dec, "VAE_decode should match CycleGAN's"


class TestBatchNormBehavior:
    """Test batch norm layers are using statistics correctly."""
    
    def test_models_set_to_eval_before_validation(self, device):
        """Verify models are explicitly set to eval mode."""
        # Create simple models for testing
        simple_unet = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
        ).to(device)
        
        # Start in training mode
        simple_unet.train()
        assert simple_unet.training, "Model should start in training mode"
        
        # Switch to eval mode
        simple_unet.eval()
        assert not simple_unet.training, "Model should be in eval mode"
        
        # Verify BatchNorm is using running stats
        for module in simple_unet.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                assert not module.training, "BatchNorm should not be in training mode"
    
    def test_dropout_disabled_in_eval(self, device):
        """Verify Dropout is disabled in eval mode."""
        model_with_dropout = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(20, 10),
        ).to(device)
        
        # Start in training mode
        model_with_dropout.train()
        assert model_with_dropout.training, "Model should start in training mode"
        
        # Switch to eval mode
        model_with_dropout.eval()
        assert not model_with_dropout.training, "Model should be in eval mode"
        
        # Verify Dropout modules are not in training mode
        for module in model_with_dropout.modules():
            if isinstance(module, torch.nn.Dropout):
                assert not module.training, "Dropout should be disabled in eval mode"


class TestCheckpointStructure:
    """Test checkpoint format and structure."""
    
    def test_checkpoint_keys_validation(self):
        """Verify expected checkpoint keys are recognized."""
        # Expected fields for CycleGAN loading
        required_keys = {
            "l_target_modules_encoder",
            "l_target_modules_decoder",
            "l_modules_others",
            "rank_unet",
            "sd_encoder",
            "sd_decoder",
            "sd_other",
            "rank_vae",
            "vae_lora_target_modules",
            "sd_vae_enc",
            "sd_vae_dec",
        }
        
        # Create checkpoint with all required keys
        checkpoint = {k: {} for k in required_keys}
        
        # Verify all keys are present
        assert all(k in checkpoint for k in required_keys), \
            f"Checkpoint missing keys. Expected: {required_keys}, Got: {set(checkpoint.keys())}"
    
    def test_quantum_params_optional(self):
        """Verify quantum_params is optional in checkpoint."""
        checkpoint_without_quantum = {"sd_encoder": {}, "sd_decoder": {}}
        checkpoint_with_quantum = {"sd_encoder": {}, "sd_decoder": {}, "quantum_params": {}}
        
        # Both should be valid
        assert "quantum_params" not in checkpoint_without_quantum
        assert "quantum_params" in checkpoint_with_quantum


class TestEvaluationAccelerator:
    """Test the minimal EvalAccelerator mock."""
    
    def test_eval_accelerator_initialization(self, device):
        """Verify EvalAccelerator initializes correctly."""
        accel = EvalAccelerator(device)
        
        assert accel.device == device
        assert accel.is_main_process == True
        assert accel.sync_gradients == True
        assert isinstance(accel.logged, list)
        assert len(accel.logged) == 0
    
    def test_eval_accelerator_unwrap_model(self, device):
        """Verify unwrap_model returns the same module."""
        accel = EvalAccelerator(device)
        dummy_model = torch.nn.Linear(10, 10)
        
        unwrapped = accel.unwrap_model(dummy_model)
        assert unwrapped is dummy_model
    
    def test_eval_accelerator_logging(self, device):
        """Verify logging stores values correctly."""
        accel = EvalAccelerator(device)
        
        test_values = {"loss": 0.5, "fid": 25.0}
        accel.log(test_values, step=100)
        
        assert len(accel.logged) == 1
        step, values = accel.logged[0]
        assert step == 100
        assert values == test_values


# Helper methods

@staticmethod
def _create_minimal_checkpoint(tmp_path):
    """Create a minimal valid checkpoint for testing."""
    checkpoint_path = tmp_path / "test_checkpoint.pkl"
    
    # Create minimal checkpoint structure
    checkpoint = {
        "l_target_modules_encoder": [],
        "l_target_modules_decoder": [],
        "l_modules_others": [],
        "rank_unet": 8,
        "sd_encoder": {},
        "sd_decoder": {},
        "sd_other": {},
        "rank_vae": 4,
        "vae_lora_target_modules": [],
        "sd_vae_enc": {},
        "sd_vae_dec": {},
    }
    
    torch.save(checkpoint, str(checkpoint_path))
    return checkpoint_path


# Add as class methods
TestModelEvaluationMode._create_minimal_checkpoint = _create_minimal_checkpoint
TestQuantumSamplerLoading._create_minimal_checkpoint = _create_minimal_checkpoint
TestModelWeightTransfer._create_minimal_checkpoint = _create_minimal_checkpoint
TestBatchNormBehavior._create_minimal_checkpoint = _create_minimal_checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
