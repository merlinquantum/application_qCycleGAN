import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import perceval as pcvl
    from merlin import QuantumLayer, ComputationSpace, LexGrouping
except Exception:  # pragma: no cover - handled via tests/mocks
    pcvl = None
    QuantumLayer = None
    ComputationSpace = None

from config import QuantumConfig


AMPLITUDE_ENCODING = True


class BosonSampler(nn.Module):
    """
    Basic (trainable) boson sampler.

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
        
        # Get quantum dimensions from config (don't compute from dims)
        self.m = self.config.num_modes
        self.n = self.config.num_photons
        
        self.config.validate()
        self.eps = self.config.epsilon
        output_size = math.prod(self.dims)
        self.input_dim = math.comb(self.m, self.n)

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
        input_state = [1] * self.n + [0] * (self.m - self.n)
        if self.use_haar_unitary:
            trainable_parameters = []
        else:
            trainable_parameters = list(self.config.trainable_parameters or ["phi"])
        computation_space = getattr(ComputationSpace, self.config.computation_space, ComputationSpace.UNBUNCHED)
        self.input_size = math.prod(self.dims)
        self.amplitude_encoding = AMPLITUDE_ENCODING

        self.model = QuantumLayer(
            circuit=self.circuit,
            trainable_parameters=trainable_parameters,
            n_photons = self.n,
            device=self.device,
            dtype=torch.float32,
            computation_space=computation_space,
            amplitude_encoding=self.amplitude_encoding,
        )
        logger.info(
            "[Quantum Encoder unsorted] Using {} modes and {} photons",
            self.m,
            self.n,
        )
        # For mocks/testing: ensure model has correct input and output sizes
        # In real merlin, these are computed automatically from circuit/n_photons
        hilbert_dim = math.comb(self.m, self.n)
        if not hasattr(self.model, 'input_size') or self.model.input_size == 0:
            self.model.input_size = hilbert_dim
        if not hasattr(self.model, 'output_size') or self.model.output_size == 64:
            self.model.output_size = hilbert_dim

        # Add linear layer as an alternative to grouping
        #self.linear = nn.Linear(self.model.output_size, output_size)
        self.grouping = LexGrouping(self.model.output_size, output_size)


    @classmethod
    def from_config(cls, dims: Sequence[int], config: QuantumConfig, **kwargs) -> "BosonSampler":
        return cls(dims=dims, config=config, **kwargs)

    def _build_circuit(self):
        if pcvl is None:
            raise ImportError("perceval is required to build BosonSampler circuits")
        if self.use_haar_unitary:
            unitary = self.unitary or pcvl.Matrix.random_unitary(self.m)
            circuit = pcvl.Circuit(self.m)
            circuit.add(0, pcvl.components.Unitary(unitary))
            logger.info(
            "[Quantum Encoder unsorted] Using Haar matrix on {} modes ({} photons)",
            self.m,
            self.n,
        )
            return circuit
        return pcvl.components.GenericInterferometer(
            self.m,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

    def _prepare_amplitudes(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize classical data into amplitude-encoding form."""
        flat = x.view(x.shape[0], -1)
        min_vals = flat.min(dim=1, keepdim=True).values
        max_vals = flat.max(dim=1, keepdim=True).values
        denom = (max_vals - min_vals).clamp_min(self.eps)
        normalized = (flat - min_vals) / denom
        normalized = normalized.clamp_(0.0, 1.0)
        amplitudes = torch.sqrt(normalized + self.eps)
        norm = torch.linalg.vector_norm(amplitudes, dim=1, keepdim=True).clamp_min(self.eps)
        amplitudes = amplitudes / norm
        current_dim = amplitudes.shape[1]
        if current_dim < self.input_dim:
            #print(f"\n [Quantum Encoder] Padding input from {current_dim} to {self.input_dim} dimensions.\n")
            pad_width = self.input_dim - current_dim
            amplitudes = F.pad(amplitudes, (0, pad_width))
        elif current_dim > self.input_dim:
            #print(f"\n [Quantum Encoder] Truncating input from {current_dim} to {self.input_dim} dimensions.\n")
            amplitudes = amplitudes[:, : self.input_dim]
        return amplitudes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)
        amplitudes = self._prepare_amplitudes(x)
        quantum_out = self.model(amplitudes)
        return self.grouping(quantum_out)

    def to(self, device):
        """Move the sampler to another device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        #self.linear = self.linear.to(device=self.device)
        self.grouping = self.grouping.to(device=self.device)
        self.model.computation_process.simulation_graph = (
            self.model.computation_process.simulation_graph.to(self.device)
        )
        return self
