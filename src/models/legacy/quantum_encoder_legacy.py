import math
import numpy
import numpy as np
import time
import torch
import os
import perceval as pcvl
from itertools import product
from typing import Tuple, List, Union
import multiprocessing as mp
import perceval as pcvl  # Import Perceval only in worker process
import exqalibur as xqlbr

try:
    from . import slos_hack as _slos_hack
except ImportError:  # pragma: no cover - allows running outside package context
    import slos_hack as _slos_hack

allstate_iterator = _slos_hack.allstate_iterator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class QEncoder:
    """
    Encode a N_1 x N_2 x ... x N_k real tensor into a Fock state

    Attributes:
    -----------
    dims : tuple
        Dimensions of the tensor to encode (e.g. (N_1, N_2, ..., N_k))
    m : int
        Number of modes in the circuit.
    n : int
        Number of photons in the circuit.
    initial_state : BasicState
        Initial state of the circuit
    postselected_states : list(BasicStates)
        List of the Fock states use to decode the values of the tensor
    value_indices : list(int)
        Given the list postselected_states, gives the indices of the states which correspond to [|1|] states
    value_labels : list(tuple(int))
        For each of the indices in value_indices, returns the number of photons in each position-modes

    Methods:
    --------
    get_post_selected_space():
        Generates the set of states/indices/labels which are use to decode the Fock State
    encode(data):
        Returns the phases associated with the tensor
    decode_from_probs(output):
        Decodes the tensor for a single probability distribution
    """

    def __init__(self, dims: list | tuple) -> None:
        """
        Constructs all the necessary attributes for the QEncoder object.

        Parameters:
        -----------
        dims : tuple(int)
            Dimensions of the tensor
        """
        self.dims = dims
        self.m = sum(self.dims) + 2
        self.n = len(self.dims) + 1

        self.get_post_selected_space()

    def get_post_selected_space(self):
        ranges = [range(dim) for dim in self.dims] + [range(2)]
        self.postselected_states = []
        self.postselected_states_idx = []
        index = 0
        all_outputs_idx_map = {state: i for i, state in
                               enumerate(allstate_iterator(xqlbr.FockState([self.n] + [0] * (self.m - 1))))}
        self.value_indices = []
        self.value_labels = []
        for state in product(*ranges):
            fock_state = []
            for dim_index in range(len(self.dims)):
                fock_state += [int(ii == state[dim_index]) for ii in range(self.dims[dim_index])]
            fock_state += [int(x == state[-1]) for x in range(2)]
            self.postselected_states += [pcvl.BasicState(fock_state)]
            self.postselected_states_idx += [all_outputs_idx_map[xqlbr.FockState(fock_state)]]
            if state[-1] == 0:
                self.value_indices += [index]
                self.value_labels += [state[:-1]]
            index += 1

    def _encode(self, data: torch.Tensor):
        sv = pcvl.StateVector()
        data = data.flatten()
        positive_amplitudes = torch.sqrt(data) / math.sqrt(len(data))
        negative_amplitudes = torch.sqrt(1 - data) / math.sqrt(len(data))
        for state_index, state in enumerate(self.postselected_states):
            if state_index % 2 == 0:  # |position, 1, 0>
                sv += float(positive_amplitudes[int(state_index / 2)]) * state
            else:  # |position, 0, 1>
                sv += float(negative_amplitudes[int((state_index - 1) / 2)]) * state
        return sv

    def encode(self, data: torch.Tensor):
        """Generates an input state vector."""
        if len(data.shape) == len(self.dims):
            assert data.shape == self.dims
            return self._encode(data)
        elif len(data.shape) == len(self.dims) + 1:
            assert data[0, ...].shape == self.dims
            state_vectors = []
            for index in range(len(data)):
                state_vectors += [self._encode(data[index])]
            return state_vectors

    def _decode_nparray(self, output: numpy.ndarray):
        all_probabilities = np.array([output[idx] for idx in self.postselected_states_idx], dtype=float)

        even_elements = all_probabilities[::2]
        odd_elements = all_probabilities[1::2]

        # Calculate the desired array
        return torch.Tensor(even_elements / (even_elements + odd_elements)).reshape(self.dims)

    def _decode_bsdist(self, output: dict):
        """
        Decode the output of a circuit from a probability distribution(s) of the form:
        { BasiceState1: p1,
          BasiceState2: p2,
          ...
        }
        The decoded value is given as the conditional probability: v = P[v=1 | position]

        NOTE: This is adapted from some other code, I haven't checked if it works

        Parameters:
        -----------
        dims : dict
            Probability distribution

        Returns:
        --------
        list
            Decoded values
        """
        all_probabilities = [output[state] for state in self.postselected_states]

        values = [all_probabilities[ii] / (all_probabilities[ii] + all_probabilities[ii + 1]) for ii in
                  self.value_indices]
        return torch.Tensor(values).reshape(self.dims)

    def decode(self, output: numpy.ndarray | dict | list) -> torch.Tensor:
        if isinstance(output, np.ndarray):
            return self._decode_nparray(output)
        elif isinstance(output, dict):
            return self._decode_bsdist(output)
        else:
            outputs = torch.stack([self._decode_nparray(o) for o in output])
            return outputs


class BosonSampler:
    """
    Basic (trainable) boson sampler.

    Attributes:
    -----------
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
    """

    def __init__(self, m: int, n: int, circuit: None | pcvl.Circuit = None) -> None:
        """
        Constructs all the necessary attributes for the VariationalCircuit object.

        Parameters:
        -----------
        m : int
            Number of modes in the circuit.
        n : int
            Number of photons in the circuit.
        circuit (optional): pcvl.Circuit
            Circuit to implement
        """
        self.m = m
        self.n = n

        if circuit is None:
            self.circuit = pcvl.components.GenericInterferometer(
                self.m,
                pcvl.components.catalog['mzi phase last'].generate,
                shape=pcvl.InterferometerShape.RECTANGLE
            )
        else:
            self.circuit = circuit
        self.parameters = self.circuit.get_parameters()
        self.n_params = len(self.parameters)
        self.init_phases()
        self.unitary = self.circuit.compute_unitary(use_symbolic=False)

        self.comb = math.comb(self.m + self.n - 1, self.n)
        self.slos = pcvl.SLOSBackend()
        self.slos.set_circuit(pcvl.components.Unitary(self.unitary))

        # precompute the normalization factors
        input_state = xqlbr.FockState([self.n] + [0] * (self.m - 1))
        self.norm = np.zeros((self.comb,), dtype=np.complex128)
        for i, state in enumerate(allstate_iterator(input_state)):
            self.norm[i] = np.sqrt(state.prodnfact())

    def init_phases(self):
        """Initializes the phases of the parameters to pi."""
        for p in self.parameters:
            # p.set_value(np.pi)
            p.set_value(2 * np.pi * np.random.rand())

    def set_parameters(self, params):
        """
        Sets the parameters of the variational circuit.

        Parameters:
        -----------
        params : dict
            Dictionary of parameter names and their values.
        """
        for p in self.parameters:
            p.set_value(params[p.name])

        # change slos cache only if changing parameters
        self.unitary = self.circuit.compute_unitary(use_symbolic=False)
        self.slos.set_circuit(pcvl.components.Unitary(self.unitary))

    def _compute(self, input_state: pcvl.StateVector, return_array=True):
        upa = np.zeros(self.comb, dtype=np.complex128)
        for fock_state, prob_ampli in input_state:
            upa += self.slos.all_unnormalized_probampli(fock_state) * prob_ampli

        # We can normalize at the end because the input always has 1 photon per mode
        upa = np.multiply(upa, self.norm)

        if return_array:
            return abs(upa) ** 2

        probs = pcvl.BSDistribution()
        for output_state, probampli in zip(allstate_iterator(self.slos._input_state), upa):
            probs[output_state] = abs(probampli) ** 2
        return probs

    def compute(self, input_states: list | pcvl.StateVector, return_array=True):
        """
        Computes the evolution of a set of input states.

        Parameters:
        -----------
        input_states : array-like
            Input data to the circuit.

        Returns:
        --------
        list(dict)
            The output probability distributions (one for each input)
        """

        if isinstance(input_states, list):
            # Collect the outputs
            output = []

            for datapoint in input_states:
                probs = self._compute(datapoint)
                output += [probs]
        else:
            print(f"-- Input_states = {input_states.shape}")
            output = self._compute(input_states, return_array=return_array)
        return output

    def load_from_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        q_params = ckpt['quantum_params']
        self.set_parameters(q_params)


def init_worker(dims: Tuple[int, ...], unitary) -> None:
    """Initialize worker process with QEncoder and BosonSampler instances."""
    global qencoder, bs
    # Consider making seed configurable
    qencoder = QEncoder(dims)
    bs = BosonSampler(qencoder.m, qencoder.n, pcvl.Unitary(unitary))


def encode(data: torch.Tensor) -> torch.Tensor:
    """Encode and evolve quantum data in worker process."""
    encoded_data = qencoder.encode(data)
    evolved_data = bs.compute(encoded_data, return_array=True)
    return qencoder.decode(evolved_data)


class ParallelQuantumEncoder:
    """Parallel implementation of quantum encoding using multiprocessing."""

    def __init__(self, dims: Tuple[int, ...], num_processes: int = None) -> None:
        """
        Initialize parallel encoder with specified dimensions and process count.

        Args:
            dims: Dimensions of the input tensor (excluding batch dimension)
            num_processes: Number of processes to use. Defaults to CPU count if None.
        """
        self.dims = dims
        self.num_processes = num_processes or mp.cpu_count()
        np.random.seed(42)
        self.qencoder = QEncoder(dims)
        self.bs = BosonSampler(self.qencoder.m, self.qencoder.n)

        self.pool = mp.Pool(
            processes=self.num_processes,
            initializer=init_worker,
            initargs=(dims, self.bs.unitary.tonp())
        )

    def _split_batch(self, batch: torch.Tensor, num_chunks: int) -> list[torch.Tensor]:
        """Split a batched tensor into chunks for parallel processing."""
        return torch.chunk(batch, num_chunks, dim=0)

    def encode(self, batch_data: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Encode a batched tensor in parallel.

        Args:
            batch_data: Tensor of shape (batch_size, *dims) to encode

        Returns:
            Tensor: Encoded results with same batch size
        """
        if isinstance(batch_data, list):
            # Handle list of tensors
            results = self.pool.map_async(encode, batch_data)
            return torch.stack(results.get())

        if not isinstance(batch_data, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if batch_data.dim() != len(self.dims) + 1:
            raise ValueError(f"Expected input of shape (batch_size, {self.dims}), "
                             f"got shape {tuple(batch_data.shape)}")

        try:
            # Determine optimal chunk size based on batch size and number of processes
            batch_size = batch_data.shape[0]
            num_chunks = min(batch_size, self.num_processes)
            # Split the batch into chunks
            chunks = self._split_batch(batch_data, num_chunks)
            # Process chunks in parallel
            results = self.pool.map_async(encode, chunks)
            # Concatenate results along batch dimension
            return torch.cat(results.get(), dim=0)

        except Exception as e:
            self.pool.terminate()
            raise RuntimeError(f"Parallel encoding failed: {str(e)}") from e

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()


if __name__ == '__main__':
    """ Example of a use of the encoder + (generic) boson sampler"""
    dims = (4, 16, 16)
    num_process = 1

    # Set the random seeds
    torch.manual_seed(42)

    data_batch = torch.rand(8, 4, 16, 16)
    pqencoder = ParallelQuantumEncoder(dims, num_processes=5)
    bs = pqencoder.bs
    print("-- Encoder defined --")
    for _ in range(10):
        start_time = time.time()
        encoded_data_batch = pqencoder.encode(data_batch)

        print(f"Encoded data batch of dim {encoded_data_batch}")
        encoded_data_batch = list(encoded_data_batch)
        print(f"Encoded data batch of len {len(encoded_data_batch)}")
        quantum_embs = bs.compute(encoded_data_batch, return_array=True)
        print(f"Quantum_embs of dim {quantum_embs.shape}")
        decoded = pqencoder.decode(quantum_embs)
        # apply logit to transfer to correct domain
        decoded = torch.logit(decoded)
        print("Time:", time.time() - start_time)
        print(decoded.shape)
