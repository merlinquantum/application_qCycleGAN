
import math
import time
from itertools import combinations, product
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:  # pragma: no cover - loaded at runtime
    import perceval as pcvl
except Exception:  # pragma: no cover
    pcvl = None

try:  # pragma: no cover - loaded at runtime
    from merlin import QuantumLayer, ComputationSpace
except Exception:  # pragma: no cover
    QuantumLayer = None
    ComputationSpace = None

try:  # pragma: no cover - loaded at runtime
    import exqalibur as xqlbr
except Exception:  # pragma: no cover
    xqlbr = None

try:  # pragma: no cover - loaded at runtime
    from perceval.utils import allstate_iterator as pcvl_allstate_iterator
except Exception:  # pragma: no cover
    pcvl_allstate_iterator = None

try:  # pragma: no cover - loaded at runtime
    from .legacy.slos_hack import allstate_iterator as slos_allstate_iterator
except Exception:  # pragma: no cover
    try:
        from slos_hack import allstate_iterator as slos_allstate_iterator
    except Exception:  # pragma: no cover
        slos_allstate_iterator = None

from config import QuantumConfig
from .quantum_encoder_unsorted import BosonSampler as BosonSamplerUnsorted

AMPLITUDE_ENCODING = True
#TODO: 
# - HUGE MEMORY ISSUE WITH SORTED ENCODING
# - possible reason: using 38 modes and 4 photons results in a space of dimension 73k
# - this leads to memory issues (even on the H100 because it is simply too big)
# - most of these 73k values are 0
# - one solution could be to support sparse tensors in MerLin
# - meanwhile, we can use unsorted encoding which is way less memory hungry and we can use 20 modes / 3 photons

class _SortedBosonSampler(nn.Module):
    """
    Basic (trainable) boson sampler with encoding/post-selection (qudits-like).

    Attributes:
    -----------
    dims : tuple
        Dimensions of the vector we will pass through the model
    m : int
        Number of modes in the circuit.
    n : int
        Number of photons in the circuit.
    circuit : pcvl.Circuit
        The variational circuit.
    parameters : list
        List of parameters for the variational circuit.
    n_params : int
        Number of parameters in the variational circuit.

    Methods:
    --------
    init_phases():
        Initializes the phases of the parameters to pi.
    set_parameters(params):
        Sets the parameters of the variational circuit.
    compute(input_states):
        Computes the output distribution of the circuit.
    load_from_checkpoint(checkpoint):
        Loads the parameters from the checkpoint.
    save_checkpoint(checkpoint_path):
        Saves the parameters from the checkpoint.

    """
    def __init__(
        self,
        dims: Sequence[int],
        config: Optional[QuantumConfig] = None,
        circuit: Optional["pcvl.Circuit"] = None,
        unitary: Optional["pcvl.Matrix"] = None,
        use_haar_unitary: Optional[bool] = None,
        device: Optional[torch.device] = None,
        override_quantum_dims: bool = False,
    ) -> None:
        """
        Constructs all the necessary attributes for the VariationalCircuit object.

        Parameters:
        -----------
        dims : tuple
            Dimensions of the vector we will pass through the model
        eps : float
            Epsilon value added to avoid zero division when backpropagating.
        circuit (optional): pcvl.Circuit
            Circuit to implement
        trainable_parameters : list
            List of trainable parameters for the variational circuit. By default, all parameters are trainable
        override_quantum_dims : bool
            If True, use config num_modes and num_photons instead of computing from dims
        """
        super().__init__()
        self.config = config or QuantumConfig()
        self.dims = tuple(dims[1:]) if len(dims) >= 4 else tuple(dims)
        self.num_values = math.prod(self.dims)
        
        # Handle quantum dimension override
        if override_quantum_dims:
            if self.config.num_modes is None or self.config.num_photons is None:
                raise ValueError(
                    "override_quantum_dims=True requires num_modes and num_photons in config"
                )
            self.m = self.config.num_modes
            self.n = self.config.num_photons
        else:
            self.m = sum(self.dims) + 2
            self.n = len(self.dims) + 1
            self.config.num_modes = self.m
            self.config.num_photons = self.n
        
        # Track computation space preference (default UNBUNCHED)
        self._using_unbunched = str(getattr(self.config, "computation_space", "UNBUNCHED")).upper() == "UNBUNCHED"
        if ComputationSpace is not None:
            self.computation_space = getattr(
                ComputationSpace,
                self.config.computation_space,
                getattr(ComputationSpace, "UNBUNCHED", None),
            )
        else:  # pragma: no cover - fallback when ComputationSpace missing
            self.computation_space = None

        self.config.validate()
        self.eps = self.config.epsilon
        if self._using_unbunched:
            self.input_dim = math.comb(self.m, self.n)
        else:
            self.input_dim = math.comb(self.m + self.n - 1, self.n)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.unitary = unitary
        self.use_haar_unitary = (
            use_haar_unitary
            if use_haar_unitary is not None
            else getattr(self.config, "use_haar_unitary", False)
        )
        self.circuit = circuit or self._build_circuit()
        if self.use_haar_unitary:
            trainable_parameters = []
        else:
            trainable_parameters = list(self.config.trainable_parameters or ["phi"])
        self.amplitude_encoding = AMPLITUDE_ENCODING

        self.model = QuantumLayer(
            circuit=self.circuit,
            trainable_parameters=trainable_parameters,
            n_photons = self.n,
            device=self.device,
            dtype=torch.float32,
            computation_space=self.computation_space or getattr(ComputationSpace, "UNBUNCHED", None),
            amplitude_encoding=self.amplitude_encoding,
        )
        logger.info(
            "[Quantum Encoder sorted] Using {} modes and {} photons",
            self.m,
            self.n,
        )
        logger.info(
            "[SortedSampler] dims={}, m={}, n={}, input_dim={}, unbunched={}, device={}, amp_enc={}, trainable_params={}"
            , self.dims, self.m, self.n, self.input_dim, self._using_unbunched, self.device, self.amplitude_encoding, trainable_parameters
        )
        # Guard against Merlin builds that omit input/output metadata
        if not hasattr(self.model, 'input_size') or self.model.input_size == 0:
            self.model.input_size = self.input_dim
        if not hasattr(self.model, 'output_size') or self.model.output_size == 64:
            self.model.output_size = self.input_dim
        try:
            self.postselection_indices = self._build_postselection_indices()
        except (KeyError, ValueError, IndexError) as e:
            logger.exception("Failed building postselection indices for m=%s, n=%s", self.m, self.n)
            if override_quantum_dims:
                raise ValueError(
                    f"Invalid quantum dimensions: m={self.m}, n={self.n} cannot encode "
                    f"spatial dimensions {self.dims}. Error: {e}"
                )
            raise

    @classmethod
    def from_config(cls, dims: Sequence[int], config: QuantumConfig, **kwargs) -> "BosonSampler":
        """Instantiate a sampler directly from a validated `QuantumConfig`."""
        return cls(dims=dims, config=config, **kwargs)

    def _build_circuit(self):
        """Create the default rectangular interferometer used in training."""
        if self.use_haar_unitary:
            logger.debug("Building Haar-random unitary circuit: m=%s", self.m)
            unitary = self.unitary or pcvl.Matrix.random_unitary(self.m)
            circuit = pcvl.Circuit(self.m)
            circuit.add(0, pcvl.components.Unitary(unitary))
            return circuit
        logger.debug("Building circuit: m=%s, shape=RECTANGLE", self.m)
        return pcvl.components.GenericInterferometer(
            self.m,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

    def _enumerate_states_fallback(self):
        """
        Generate the relevant Fock basis when the native iterator is unavailable.

        When the computation space is UNBUNCHED, only binary occupation (0/1)
        states are valid. Otherwise we fall back to enumerating all integer
        compositions of n photons over m modes.
        """
        if self._using_unbunched:
            for combo in combinations(range(self.m), self.n):
                state = [0] * self.m
                for idx in combo:
                    state[idx] = 1
                yield tuple(state)
            return

        state = [0] * self.m

        def backtrack(position: int, remaining: int):
            if position == self.m - 1:
                state[position] = remaining
                yield tuple(state)
                return
            for value in range(remaining + 1):
                state[position] = value
                yield from backtrack(position + 1, remaining - value)

        yield from backtrack(0, self.n)

    def _build_state_index_map(self):
        """Create a mapping from Fock states to legacy indices."""
        iterator_source = None
        if self._using_unbunched:
            iterator = self._enumerate_states_fallback()
            iterator_source = "fallback-unbunched"
        else:
            iterator = None
            if xqlbr is not None:
                base_state = xqlbr.FockState([self.n] + [0] * (self.m - 1))
                if pcvl_allstate_iterator is not None:
                    iterator = pcvl_allstate_iterator(base_state)
                    iterator_source = "perceval"
                elif slos_allstate_iterator is not None:
                    iterator = slos_allstate_iterator(base_state)
                    iterator_source = "slos_hack"
            if iterator is None:
                logger.warning("Using fallback state enumerator (unbunched=%s): native iterators unavailable", self._using_unbunched)
                iterator = self._enumerate_states_fallback()
                iterator_source = iterator_source or "fallback"

        state_index = {tuple(state): idx for idx, state in enumerate(iterator)}
        logger.info("State index map built: %d states (iterator=%s)", len(state_index), iterator_source)
        return state_index

    def _build_postselection_indices(self):
        """
        Enumerate the |position,1,0> and |position,0,1> states used for decoding.

        These indices tell the decoder which probability entries correspond to
        the "value is 1" and "value is 0" events at each classical coordinate.
        """
        offsets = []
        running = 0
        for dim in self.dims:
            offsets.append(running)
            running += dim
        ancilla_offset = running

        state_index = self._build_state_index_map()
        positive_indices = []
        negative_indices = []
        labels = []

        axis_ranges = [range(dim) for dim in self.dims]
        for label in product(*axis_ranges):
            occupancy = [0] * self.m
            for dim_idx in range(len(self.dims)):
                occupancy[offsets[dim_idx] + label[dim_idx]] = 1
            occupancy[ancilla_offset] = 1
            pos_idx = state_index[tuple(occupancy)]
            occupancy[ancilla_offset] = 0
            occupancy[ancilla_offset + 1] = 1
            neg_idx = state_index[tuple(occupancy)]
            positive_indices.append(pos_idx)
            negative_indices.append(neg_idx)
            labels.append(label)

        self.state_labels = labels
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        old_input_dim = self.input_dim
        max_index = max(positive_indices + negative_indices) + 1 if positive_indices else self.input_dim
        if max_index > self.input_dim:
            self.input_dim = max_index
        logger.debug(
            "Postselection: %d positive / %d negative indices; labels=%d; input_dim adjusted %d->%d",
            len(positive_indices), len(negative_indices), len(labels), old_input_dim, self.input_dim,
        )
        return list(zip(positive_indices, negative_indices))

    def _encode_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert the classical tensor into a superposition matching the legacy encoder.

        Values are clamped into [0,1], converted into positive/negative ancilla
        amplitudes, and inserted into the flattened Hilbert space.
        """
        batch_size = x.shape[0]
        raw = x.view(batch_size, -1)
        raw_min = float(raw.min().item())
        raw_max = float(raw.max().item())
        flat = raw.clamp(0.0, 1.0)
        normalization = math.sqrt(flat.shape[1])
        positive = torch.sqrt(flat) / normalization
        negative = torch.sqrt(1.0 - flat) / normalization
        amplitudes = torch.zeros(batch_size, self.input_dim, device=self.device, dtype=torch.float32)
        amplitudes[:, self.positive_indices] = positive
        amplitudes[:, self.negative_indices] = negative

        logger.debug(
            "Encoding inputs: x.shape=%s -> flat.shape=%s, amplitudes.shape=%s",
            x.shape, flat.shape, amplitudes.shape,
        )
        if raw_min < 0.0 or raw_max > 1.0:
            logger.warning("Input values clamped to [0,1] (min %s, max %s)", raw_min, raw_max)

        logger.debug("Normalization sqrt(dim)=%s, amplitudes mean=%g", normalization, amplitudes.abs().mean().item())
        return amplitudes

    def _match_model_input_size(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Pad or truncate amplitudes to match the computation space size."""
        target_dim = getattr(self.model, "input_size", amplitudes.shape[1])
        current_dim = amplitudes.shape[1]
        if current_dim < target_dim:
            amplitudes = F.pad(amplitudes, (0, target_dim - current_dim))
            logger.info("Adjusting amplitudes: current_dim=%d target_dim=%d -> padded", current_dim, target_dim)
        elif current_dim > target_dim:
            amplitudes = amplitudes[:, :target_dim]
            logger.info("Adjusting amplitudes: current_dim=%d target_dim=%d -> truncated", current_dim, target_dim)
        return amplitudes

    def _decode_outputs(self, outputs: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Recover classical probabilities from grouped quantum measurement results.

        The method pairs the positive/negative ancilla outcomes, computes the
        conditional probability of the "positive" branch, and reshapes to the
        original spatial layout.
        """
        positive = outputs[:, self.positive_indices].clamp_min(0.0)
        negative = outputs[:, self.negative_indices].clamp_min(0.0)
        denom = (positive + negative).clamp_min(self.eps)
        small_denom = int((denom <= (self.eps * 10)).sum().item())
        if small_denom:
            logger.warning("Denominator small for %d entries (eps=%g)", small_denom, self.eps)
        values = positive / denom
        decoded = values.reshape((batch_size,) + self.dims)
        logger.debug("Decoded outputs: outputs.shape=%s -> decoded.shape=%s", outputs.shape, decoded.shape)
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode classical data, run it through the quantum layer, and decode it."""
        t0 = time.time()
        batch_size = x.shape[0]
        logger.debug("Forward start: batch_size=%d, input.shape=%s", batch_size, x.shape)
        x = x.to(self.device, dtype=torch.float32)
        amplitudes = self._encode_inputs(x)
        matched = self._match_model_input_size(amplitudes)
        quantum_out = self.model(matched)
        logger.debug("Quantum output shape: %s", quantum_out.shape)
        decoded = self._decode_outputs(quantum_out, batch_size)
        elapsed = time.time() - t0
        logger.debug("Forward end: decoded.shape=%s, elapsed=%.4fs", decoded.shape, elapsed)
        return decoded

    def to(self, device):
        """Move the sampler and underlying quantum layer to another device."""
        new_device = torch.device(device)
        logger.info("Moving sampler to device: %s", new_device)
        self.device = new_device
        self.model = self.model.to(self.device)
        try:
            self.model.computation_process.simulation_graph = (
                self.model.computation_process.simulation_graph.to(self.device)
            )
        except Exception:
            logger.debug("No simulation_graph to move or move failed; continuing")
        return self


class BosonSampler(nn.Module):
    """Factory wrapper choosing sorted vs unsorted encoders based on config."""

    def __new__(
        cls,
        dims: Sequence[int],
        config: Optional[QuantumConfig] = None,
        **kwargs,
    ):
        config = config or QuantumConfig()
        target_cls = _SortedBosonSampler if getattr(config, "sort_encoding", True) else BosonSamplerUnsorted
        instance = target_cls.__new__(target_cls)
        target_cls.__init__(instance, dims, config=config, **kwargs)
        return instance

    @classmethod
    def from_config(cls, dims: Sequence[int], config: QuantumConfig, **kwargs):
        return cls(dims=dims, config=config, **kwargs)


def _load_unsorted_sampler():
    from .quantum_encoder_unsorted import BosonSampler as _UnsortedBosonSampler

    return _UnsortedBosonSampler
