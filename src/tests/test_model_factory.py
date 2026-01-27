"""Tests for the model factory and checkpoint loading."""

from types import SimpleNamespace
from pathlib import Path

import torch

from config import QuantumConfig, TrainingConfig
from models import model_factory
from models.model_factory import ModelFactory, TrainableParamsManager


class DummyConvBlock(torch.nn.Module):
    """Minimal block exposing a conv weight."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1)


class DummyVAE(torch.nn.Module):
    """VAE stub with the fields accessed by the factory."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.conv_in = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.decoder = torch.nn.Module()
        self.decoder.conv_in = torch.nn.Conv2d(1, 1, kernel_size=1)
        for idx in range(1, 5):
            setattr(self.decoder, f"skip_conv_{idx}", torch.nn.Conv2d(1, 1, kernel_size=1))
        self.decoder.conv_out = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.post_quant_conv = torch.nn.Conv2d(1, 1, kernel_size=1)


class DummyCycleGAN:
    """Lightweight replacement for CycleGAN_Turbo during tests."""

    def __init__(self, accelerator=None, pretrained_path=None):
        self.pretrained_path = pretrained_path
        self.loaded_state = {}
        if pretrained_path and Path(pretrained_path).exists():
            try:
                self.loaded_state = torch.load(pretrained_path, map_location="cpu")
            except Exception:  # pragma: no cover - defensive guard for malformed stubs
                self.loaded_state = {}
        self.unet = torch.nn.Module()
        self.unet.conv_in = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.vae = DummyVAE()
        self.vae_b2a = DummyVAE()
        self.vae_enc = torch.nn.Sequential(torch.nn.Linear(1, 1))
        self.vae_dec = torch.nn.Sequential(torch.nn.Linear(1, 1))

    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a, boson_sampler=None, dynamic=False, train_full_unet=False):
        """Return a deterministic list of parameters."""
        return list(unet.parameters())


class DummyBosonSampler(torch.nn.Module):
    """Simple trainable boson sampler stand-in."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)


def _fake_initialize_unet(rank, return_lora_module_names=False):
    dummy_unet = torch.nn.Module()
    dummy_unet.conv_in = torch.nn.Conv2d(1, 1, kernel_size=1)
    modules = (["enc_mod"], ["dec_mod"], ["other_mod"])
    if return_lora_module_names:
        return dummy_unet, *modules
    return dummy_unet


def _fake_initialize_vae(rank=4, return_lora_module_names=False, dynamic=False):
    dummy_vae = DummyVAE()
    targets = ["vae_mod"]
    if return_lora_module_names:
        return dummy_vae, targets
    return dummy_vae


def test_model_factory_loads_checkpoint_with_adaptation(monkeypatch, tmp_path):
    """ModelFactory should load checkpoints and expose adaptation module state."""
    conv_ad = torch.nn.Conv2d(8, 4, kernel_size=1)
    ckpt_path = tmp_path / "model_with_adaptation.pth"
    torch.save({"conv_ad": conv_ad.state_dict()}, ckpt_path)

    # Patch heavy dependencies with lightweight stubs.
    monkeypatch.setattr(model_factory, "initialize_unet", _fake_initialize_unet)
    monkeypatch.setattr(model_factory, "initialize_vae", _fake_initialize_vae)
    monkeypatch.setattr(model_factory, "CycleGAN_Turbo", DummyCycleGAN)

    quantum_cfg = QuantumConfig(quantum=True, start_path=str(ckpt_path))
    training_cfg = TrainingConfig(hf_model_path=None)
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    components = ModelFactory.build(accelerator, training_cfg, quantum_cfg)

    assert isinstance(components.cyclegan, DummyCycleGAN)
    assert components.adaptation_module is not None, "Expected adaptation module from checkpoint"
    assert components.adaptation_module.weight.shape[0] == 4

    params = TrainableParamsManager.trainable_params(components, quantum_cfg)
    assert any(p is components.adaptation_module.weight for p in params), "Adaptation module should be trainable"


def test_model_factory_loads_real_checkpoint(monkeypatch):
    """Ensure we can load the bundled classical checkpoint (model_1501.pkl)."""
    ckpt_path = Path(__file__).parent / "model_1501.pkl"
    assert ckpt_path.exists(), "Expected bundled checkpoint file to exist."

    monkeypatch.setattr(model_factory, "initialize_unet", _fake_initialize_unet)
    monkeypatch.setattr(model_factory, "initialize_vae", _fake_initialize_vae)
    monkeypatch.setattr(model_factory, "CycleGAN_Turbo", DummyCycleGAN)

    quantum_cfg = QuantumConfig(quantum=True, start_path=str(ckpt_path))
    training_cfg = TrainingConfig(hf_model_path=None)
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    components = ModelFactory.build(accelerator, training_cfg, quantum_cfg)

    assert isinstance(components.cyclegan, DummyCycleGAN)
    assert components.adaptation_module is None
    params = TrainableParamsManager.trainable_params(components, quantum_cfg)
    assert isinstance(params, list)
    # ensure key tensors exist in state dict
    sd = torch.load(ckpt_path, map_location="cpu")
    assert "sd_encoder" in sd and "sd_decoder" in sd

    classical_components = ModelFactory.build(
        accelerator,
        training_cfg,
        QuantumConfig(quantum=False, start_path=str(ckpt_path)),
    )

    assert classical_components.cyclegan.loaded_state.keys() == components.cyclegan.loaded_state.keys()
    assert classical_components.cyclegan.loaded_state["sd_encoder"].keys() == components.cyclegan.loaded_state["sd_encoder"].keys()

    for key in list(sd["sd_encoder"].keys())[:5]:
        ref = sd["sd_encoder"][key]
        assert torch.equal(classical_components.cyclegan.loaded_state["sd_encoder"][key], ref)
        assert torch.equal(components.cyclegan.loaded_state["sd_encoder"][key], ref)


def test_model_factory_pretrained_classical_then_quantum(monkeypatch):
    """Loading classical weights should still allow quantum training with Boson sampler."""
    ckpt_path = Path(__file__).parent / "model_1501.pkl"
    assert ckpt_path.exists()

    monkeypatch.setattr(model_factory, "initialize_unet", _fake_initialize_unet)
    monkeypatch.setattr(model_factory, "initialize_vae", _fake_initialize_vae)
    monkeypatch.setattr(model_factory, "CycleGAN_Turbo", DummyCycleGAN)

    quantum_cfg = QuantumConfig(quantum=True, start_path=str(ckpt_path))
    training_cfg = TrainingConfig(hf_model_path=None)
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    components = ModelFactory.build(accelerator, training_cfg, quantum_cfg)
    boson = DummyBosonSampler()

    params = TrainableParamsManager.trainable_params(components, quantum_cfg, boson_sampler=boson)
    assert any(p is boson.linear.weight for p in params)
    assert any(p is boson.linear.bias for p in params)

    sd = torch.load(ckpt_path, map_location="cpu")
    sample_key = next(iter(sd["sd_encoder"]))
    assert torch.equal(components.cyclegan.loaded_state["sd_encoder"][sample_key], sd["sd_encoder"][sample_key])

# additional validation done when running a model with pretrained path : OK


def test_freeze_vae_encoder_keeps_decoder_io_and_skip_convs_trainable():
    """freeze_vae_encoder should freeze the encoder and keep decoder IO + skip convs trainable."""
    dummy_vae_a2b = DummyVAE()
    dummy_vae_b2a = DummyVAE()
    components = model_factory.ModelComponents(
        unet=torch.nn.Module(),
        vae_enc=torch.nn.Module(),
        vae_dec=torch.nn.Module(),
        vae_a2b=dummy_vae_a2b,
        vae_b2a=dummy_vae_b2a,
        lora_modules_encoder=[],
        lora_modules_decoder=[],
        lora_modules_others=[],
        vae_lora_target_modules=[],
        cyclegan=None,
    )

    TrainableParamsManager.freeze_vae_encoder(components)

    for vae in (dummy_vae_a2b, dummy_vae_b2a):
        assert not any(p.requires_grad for p in vae.encoder.parameters())
        assert any(p.requires_grad for p in vae.post_quant_conv.parameters())
        for idx in range(1, 5):
            skip = getattr(vae.decoder, f"skip_conv_{idx}")
            assert any(p.requires_grad for p in skip.parameters())
        assert any(p.requires_grad for p in vae.decoder.conv_in.parameters())
        assert any(p.requires_grad for p in vae.decoder.conv_out.parameters())

        # Ensure only the intended decoder pieces are trainable for this dummy VAE.
        trainable = [
            name
            for name, p in vae.named_parameters()
            if p.requires_grad
        ]
        allowed_prefixes = (
            "decoder.conv_in.",
            "decoder.conv_out.",
            "decoder.skip_conv_1.",
            "decoder.skip_conv_2.",
            "decoder.skip_conv_3.",
            "decoder.skip_conv_4.",
            "post_quant_conv.",
        )
        assert all(name.startswith(allowed_prefixes) for name in trainable)
