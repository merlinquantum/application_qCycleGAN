# Quantum-Enhanced CycleGAN Turbo

This directory contains the refactored training and evaluation codebase for the Airbus/BMW Hybrid CycleGAN-Turbo project. It integrates quantum-enhanced image-to-image translation using Boson samplers with the classical CycleGAN-Turbo framework. The **classical component** for this hybrid model is a diffusion-based image translation network (CycleGAN-Turbo) trained with LoRA adapters, from https://github.com/GaParmar/

## Quick Start

### 1. Training Your First Model

The simplest way to begin is to run the default configuration:

```bash
python train.py --tracker_project_name my_wandb_project --output_dir ./outputs/run001
```

This will:
- Load the default configuration from `config/experiments/default.yaml`
- Instantiate a quantum-enhanced CycleGAN (Boson sampler + LoRA fine-tuning)
- Train on the specified dataset for 25,000 steps (or less)
- Log metrics to Weights & Biases

### 2. Running Classical (Non-Quantum) Mode

To disable the Boson sampler and run pure classical LoRA fine-tuning:

```bash
python train.py --quantum false --tracker_project_name my_wandb_project
```

### 3. Quick Validation (GPU Smoke Test)

To verify your environment and GPU setup with a tiny dataset:

```bash
python train.py --experiment_config config/experiments/gpu_smoke.yaml \
  --tracker_project_name smoke_test --output_dir ./outputs/smoke
```

This runs 5 training steps on a minimal dataset, perfect for CI/CD pipelines or debugging.

## Project Structure

```
src_quantum/
├── README.md                      # This file
├── train.py                       # **Main entry point** for training
├── get_metrics.py                 # Evaluation & metric computation (FID, DINO)
├── verify_quantum_training.py     # macOS environment sanity check
│
├── config/                        # Configuration system (dataclasses + YAML)
│   ├── defaults.py                # QuantumConfig, DatasetConfig, TrainingConfig
│   ├── loader.py                  # load_configs() helper for YAML parsing
│   └── experiments/               # Pre-built experiment bundles
│       ├── default.yaml           # Full quantum pipeline baseline
│       ├── classical_run.yaml     # Pure classical LoRA (Day↔Night)
│       └── gpu_smoke.yaml         # Tiny dataset for quick tests
│
├── models/                        # Neural network modules & builders
│   ├── cyclegan_turbo.py          # CycleGAN-Turbo architecture (UNet + LoRA)
│   ├── quantum_encoder.py         # **Sorted Boson sampler** (default, post-selected)
│   ├── quantum_encoder_unsorted.py  # **Unsorted variant** (LexGrouping output aggregation)
│   ├── model_factory.py           # Factory for assembling UNet/VAE/Boson sampler stacks
│   └── legacy/                    # Archived implementations for reference
│
├── data/                          # Dataset utilities
│   └── (dataset wrappers, transforms, loaders)
│
├── training/                      # Training infrastructure
│   └── (Trainer class, loss functions, callbacks)
│
├── my_utils/                      # CLI helpers, device utils, argparse, etc.
│
├── tests/                         # Test suite
│   ├── test_quantum_encoder.py    # Quantum encoder validation (23 tests)
│   ├── test_config.py             # Config system tests
│   ├── test_dataset.py            # Data loading tests
│   ├── test_model_factory.py      # Model instantiation tests
│   └── test_losses.py             # Loss function tests
│
├── qCycleGAN.ipynb                # Interactive tutorial notebook
│
└── (other utilities: image_prep.py, inference_unpaired.py, etc.)
```

## Configuration System

All training parameters are managed through three config dataclasses. You can specify them via:

1. **YAML files** (`config/experiments/*.yaml`) - recommended for reproducibility
2. **CLI arguments** - useful for quick iterations
3. **Python dataclass defaults** (`config/defaults.py`)

### Key Configuration Options

#### Quantum Settings (`quantum.*`)
```yaml
quantum:
  quantum: true                    # Enable/disable Boson sampler
  sort_encoding: true              # true=sorted (default), false=unsorted (LexGrouping)
  num_modes: 20                    # M in photonic circuit
  num_photons: 3                   # N photons to use
  epsilon: 1.0e-5                  # Numerical stability parameter
  trainable_parameters: ["phi"]    # Which circuit params are trainable
```

#### Classical / Ablation Settings (`classical.*`)
```yaml
classical:
  cl_comp: false                   # Classical comparison mode (freeze VAE encoders)
  random_ablation: false           # Use random 2-layer MLP instead of Boson sampler
  random_trained: false            # Train the random layer when ablation is enabled
  random_ablation_hidden_dim: 2048 # Hidden size for the random ablation MLP
```

Set `classical.random_ablation=true` (or pass `--random_ablation true`) to swap the Boson
sampler with a simple random MLP producing a 1024-d embedding for ablation studies.
Toggle `classical.random_trained` to decide whether that layer updates during training.

#### Dataset Settings (`dataset.*`)
```yaml
dataset:
  dataset_folder: ../data/dataset_full_scale/
  train_img_prep: no_resize        # Image preprocessing strategy
  train_batch_size: 1              # Batch size for training
  validation_num_images: 4         # Images to generate per validation
```

#### Training Settings (`training.*`)
```yaml
training:
  learning_rate: 1.0e-5
  max_train_steps: 25000
  checkpointing_steps: 5000        # Save model checkpoint every N steps
  validation_steps: 2000           # Validate every N steps
  tracker_project_name: my_project # W&B project name
```

## Parameters and Modules Trained

Training behavior is controlled primarily by `quantum.quantum`, `classical.cl_comp`, and `classical.random_ablation`
(note: `classical.random_ablation=true` is incompatible with `quantum.quantum=true`).

### When `quantum.quantum=true` OR `classical.cl_comp=true` OR `classical.random_ablation=true`

- **VAE encoder**: frozen (including `quant_conv` when present).
- **VAE decoder pipeline**: kept the same; trained via LoRA adapters (`vae_skip`).
- **Fully-trained VAE components**: `decoder.conv_in`, `decoder.conv_out`, `decoder.skip_conv_{1..4}`, `post_quant_conv`.
- **UNet**:
  - If `training.unet_trained=true`: all UNet parameters train.
  - Else: UNet trains via LoRA adapters + `unet.conv_in`.
- **Sampler head**:
  - `quantum.quantum=true`: Boson sampler trains if `quantum.quantum_trained=true`.
  - `classical.random_ablation=true`: random MLP trains if `classical.random_trained=true`.

### When none of them are enabled (standard training)

- **VAE**: encoder + decoder are trainable through the configured LoRA adapter (`vae_skip`) and the added decoder skip-convs.
- **UNet**: same rule as above for `training.unet_trained` (full UNet vs LoRA + `unet.conv_in`).

### CLI Overrides

Any config field can be overridden from the command line:

```bash
python train.py \
  --experiment_config config/experiments/default.yaml \
  --quantum false \                       # Override quantum setting
  --train_batch_size 2 \                  # Override batch size
  --learning_rate 5.0e-5 \                # Override learning rate
  --max_train_steps 50000 \               # Override max steps
  --output_dir ./outputs/custom_run
```

## Quantum Encoder Implementations

The project provides two Boson sampler variants, selectable via `QuantumConfig.sort_encoding`:

### Sorted Encoder (Default: `sort_encoding=true`)

**File:** `models/quantum_encoder.py` (class `_SortedBosonSampler`)

**Architecture:**
1. Input amplitudes are split into positive/negative pairs
2. Each spatial coordinate is post-selected as a quantum measurement outcome
3. Conditional probabilities are computed via Fock state projections

**When to use:** 
- Default choice for all new experiments
- Compatible with all existing checkpoints
- More sophisticated quantum encoding scheme

**Hyperparameters:**
- `num_modes (M)`: Total modes in the interferometer
- `num_photons (N)`: Photon number constraint
- `sort_encoding=true`: Activates this variant, when the amplitude encoding is not done with respect to each selected Fock state (less memory needed)

### Unsorted Encoder (`sort_encoding=false`)

**File:** `models/quantum_encoder_unsorted.py` (class `BosonSampler`)

**Architecture:**
1. Input amplitudes are directly encoded into the quantum circuit
2. Output Fock state probabilities are aggregated via Merlin's `LexGrouping` strategy
3. Simpler, parameter-efficient alternative

**When to use:**
- Faster experiments (fewer quantum evaluations)
- Research exploring different output mapping strategies
- Parameter-constrained scenarios

**Hyperparameters:**
- `num_modes (M)`: Must be provided in config (not computed from dims)
- `num_photons (N)`: Must be provided in config
- `sort_encoding=false`: Activates this variant

**Switching Between Variants:**

In your YAML config:
```yaml
quantum:
  sort_encoding: true     # Use sorted (default, high memory demand)
  # or
  sort_encoding: false    # Use unsorted (LexGrouping)
```

Both variants expose the same PyTorch `nn.Module` interface, so switching is seamless.

## Training Workflow

### Phase 1: Preparation
```bash
# Verify your environment
python verify_quantum_training.py

# Explore the available configs
ls config/experiments/
```

### Phase 2: Launch Training
```bash
python train.py \
  --experiment_config config/experiments/default.yaml \
  --tracker_project_name my_project \
  --output_dir ./outputs/experiment_001
```

### Phase 3: Monitoring
- **Weights & Biases:** View real-time metrics at `https://wandb.ai/your-user/my_project`
- **Local checkpoints:** Saved to `./outputs/experiment_001/` every `checkpointing_steps`

### Phase 4: Evaluation
```bash
python get_metrics.py \
  --checkpoint_dir ./outputs/experiment_001/checkpoint-25000 \
  --quantum true \
  --output_dir ./outputs/experiment_001/metrics
```

This computes FID and DINO metrics on validation images.

## Examples

### Example 1: Run Classical LoRA (No Quantum)
```bash
python train.py \
  --experiment_config config/experiments/classical_run.yaml \
  --tracker_project_name classical_baseline \
  --quantum false
```

### Example 2: Quantum Run with Custom Hyperparameters
```bash
python train.py \
  --quantum true \
  --num_modes 16 \
  --num_photons 2 \
  --sort_encoding true \
  --train_batch_size 4 \
  --learning_rate 2.0e-5 \
  --max_train_steps 50000 \
  --tracker_project_name quantum_custom \
  --output_dir ./outputs/quantum_v2
```

### Example 3: Quick Smoke Test
```bash
python train.py \
  --experiment_config config/experiments/gpu_smoke.yaml \
  --tracker_project_name smoke_test
```

## Testing

The project includes a comprehensive test suite (23 tests) covering:
- Quantum encoder implementations (sorted & unsorted)
- Configuration loading and validation
- Dataset utilities
- Model factory instantiation

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_quantum_encoder.py -v
```

For detailed testing information, see `TESTING.md`.

## Extending the Project

### Adding a New Experiment Config

1. Create a new YAML file in `config/experiments/my_new_experiment.yaml`
2. Copy the structure from an existing config (e.g., `default.yaml`)
3. Customize the `quantum`, `dataset`, and `training` sections
4. Run: `python train.py --experiment_config config/experiments/my_new_experiment.yaml`

### Modifying Default Hyperparameters

Edit `config/defaults.py` to change base values for all configs:
```python
# config/defaults.py
@dataclass
class QuantumConfig:
    quantum: bool = True
    num_modes: int = 20        # Change this default
    num_photons: int = 3       # or this
    # ...
```

### Adding a New Model Component

1. Implement your module as a PyTorch `nn.Module`
2. Update `models/model_factory.py` to instantiate it
3. Add unit tests in `tests/` (following existing patterns)
4. Document the new component in this README

## Key Files Reference

| File | Purpose |
|------|---------|
| `train.py` | Main training entry point |
| `config/defaults.py` | Configuration dataclass definitions |
| `config/loader.py` | YAML config parsing |
| `models/quantum_encoder.py` | Sorted Boson sampler (default) |
| `models/quantum_encoder_unsorted.py` | Unsorted Boson sampler (LexGrouping) |
| `models/model_factory.py` | Model instantiation & component factory |
| `get_metrics.py` | FID & DINO metric computation |
| `qCycleGAN.ipynb` | Interactive tutorial & demo |
| `tests/test_quantum_encoder.py` | Quantum encoder test suite (23 tests) |

## Additional Resources

- **Paper:** See `https://arxiv.org/abs/2403.12036` for the full classical CycleGAN-Turbo approach for unpaired image generation (code: https://github.com/GaParmar/img2img-turbo)
- **Tutorial:** `qCycleGAN.ipynb` provides step-by-step walkthroughs
