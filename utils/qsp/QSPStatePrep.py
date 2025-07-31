# QSPStatePrep.py
# Gabriel Waite

import cirq
import numpy as np

def controlledRySequence(ancilla_qb: cirq.Qid, workspace_qbs: list[cirq.Qid], n: int) -> cirq.Circuit:
    """
    Implements a unitary with controlled Ry rotations and a final X gate on an ancilla.

    Args:
        ancilla_qb: The ancilla qubit.
        workspace_qbs: A list of n workspace qubits (x_1, ..., x_n).
        n: The number of workspace registers. This should be equal to len(workspace_qbs).

    Returns:
        A cirq.Circuit implementing the specified unitary.
    Note:
        This is the block encoding we want
    """
    if len(workspace_qbs) != n:
        raise ValueError(f"Number of workspace qubits ({len(workspace_qbs)}) must equal n ({n}).")

    # Initialise the circuit
    circuit = cirq.Circuit()

    # Apply controlled Ry rotations
    for j in range(n):
        # cirq.ry(rads) equivalent exp(-i Y rads / 2) = cos(rads/2) I - i sin(rads/2) Y

        theta_j = (2**(j + 1)) / (2**n)

        # Apply the controlled Ry gate
        # The control is workspace_qbs[j], target is ancilla_qb
        circuit.append(cirq.ry(theta_j).on(ancilla_qb).controlled_by(workspace_qbs[j]))

    # Finally, apply a bit flip gate X to the ancilla register
    circuit.append(cirq.X(ancilla_qb))

    return circuit

class ControlledRySequenceGate(cirq.Gate):
    """
    A custom gate for the controlledRySequence circuit.
    This gate implements a controlled Ry rotation sequence followed by an X gate on an ancilla qubit.
    """

    def __init__(self, num_workspace_qubits: int):
        if num_workspace_qubits < 0:
            raise ValueError("Number of workspace qubits must be non-negative.")
        self._num_workspace_qubits = num_workspace_qubits
        self._total_qubits = 1 + num_workspace_qubits # Define total qubits.

    def _num_qubits_(self) -> int:
        return self._total_qubits

    def _decompose_(self, qubits):
        if len(qubits) != self._total_qubits:
            raise ValueError(f"Expected {self._total_qubits} qubits, but got {len(qubits)}.")

        ancilla_qb = qubits[0]
        workspace_qbs = list(qubits[1:])

        composed_circuit = controlledRySequence(
            ancilla_qb = ancilla_qb,
            workspace_qbs = workspace_qbs,
            n = self._num_workspace_qubits
        )
        yield composed_circuit.all_operations()

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return cirq.CircuitDiagramInfo(
            wire_symbols=("U",) * self._total_qubits,
            connected=True,
        )

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._num_workspace_qubits == other._num_workspace_qubits

    def __repr__(self):
        return f"ControlledRySequenceGate(num_workspace_qubits={self._num_workspace_qubits})"
    
class CRGate(cirq.Gate):
    """Implements the controlled rotation CR[control, target; phi]."""

    def __init__(self, phi: float):
        """Initializes the CRGate.

        Args:
            phi: The rotation angle in radians.
        """
        self.phi = phi

    def _num_qubits_(self) -> int:
        """Returns the number of qubits this gate acts on."""
        return 2

    def _decompose_(self, qubits):
        """Decomposes the CRGate into a list of Cirq operations.

        Args:
            qubits: A sequence of two qubits (control, target) the gate is applied to.
        Yields:
            Cirq operations that implement the CRGate.
        """
        control, target = qubits
        yield cirq.X(control)
        yield cirq.CNOT(control, target)
        yield cirq.rz(-2 * self.phi).on(target)
        yield cirq.CNOT(control, target)
        yield cirq.X(control)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        """Provides information for drawing the gate in a circuit diagram."""
        return cirq.CircuitDiagramInfo(wire_symbols=("CR","CR"), connected=True)
    
    def __eq__(self, other):
        """Determines if two CRGate instances are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.phi == other.phi

    def __hash__(self):
        """Returns a hash for the CRGate instance."""
        return hash((CRGate, self.phi))

    def __repr__(self):
        """Returns a string representation of the CRGate."""
        return f"CRGate(phi={self.phi})"
    
def StatePrep(angle_list: list, num_ws_qbs: int):
    """
    Prepares a quantum state using the given angles.
    Args:
        angle_list: List of angles for the QSP.
        num_ws_qbs: Number of workspace qubits.
    Returns:
        A cirq.Circuit that prepares the specified quantum state.
    """
    n = num_ws_qbs

    # setup circuit qubits
    qsp_anc, be_anc = cirq.NamedQubit('qsp_anc'), cirq.NamedQubit('rbe_anc')
    workspace = [cirq.NamedQubit(f'x_{i}') for i in range(1, n + 1)]

    be_qbs = [be_anc] + workspace

    # Initialise circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qsp_anc))
    circuit.append(cirq.H.on_each(workspace))

    BE_gate = ControlledRySequenceGate(num_workspace_qubits=n)

    for i in range(len(angle_list)-1,0,-1):
        print(angle_list[i])
        circuit.append(CRGate(phi=angle_list[i])(be_anc, qsp_anc))
        circuit.append(BE_gate(*be_qbs))
    # add zeroth angle
    circuit.append(CRGate(phi=angle_list[0])(be_anc, qsp_anc))

    circuit.append(cirq.H.on(qsp_anc))

    return circuit

def SVFromStatePrepCircuit(angle_list: list, num_ws_qbs: int, init_state: list[int], verbose: bool = True):
    """
    Returns a Dirac notation state vector from the state preparation circuit, given the angles and initial state.

    Args:
        angle_list: List of angles for the QSP.
        num_ws_qbs: Number of workspace qubits.
        init_state: Initial state as a list of bits (0s and 1s).

    Returns:
        A cirq.Circuit that prepares the specified quantum state.
    """
    n = num_ws_qbs

    # setup circuit qubits
    qsp_anc, be_anc = cirq.NamedQubit('qsp_anc'), cirq.NamedQubit('rbe_anc')
    workspace = [cirq.NamedQubit(f'x_{i}') for i in range(1, n + 1)]

    be_qbs = [be_anc] + workspace

    # Initialise circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(qsp_anc))
    
    # Apply initial state preparation
    for i, bit in enumerate(init_state):
        if bit == 1:
            index = n - i - 1 # Reverse the index for correct placement
            circuit.append(cirq.X.on(workspace[index]))

    BE_gate = ControlledRySequenceGate(num_workspace_qubits=n)

    for i in range(len(angle_list)-1,0,-1):
        print(angle_list[i])
        circuit.append(CRGate(phi=angle_list[i])(be_anc, qsp_anc))
        circuit.append(BE_gate(*be_qbs))
    # add zeroth angle
    circuit.append(CRGate(phi=angle_list[0])(be_anc, qsp_anc))

    circuit.append(cirq.H.on(qsp_anc))

    raw_sv = cirq.final_state_vector(circuit)
    dirac_sv = cirq.dirac_notation(raw_sv)

    if verbose:
        sigma = float(input("Enter the value of sigma: "))
        func = lambda x: np.exp(- (x**2) / (2 * sigma**2))
        x_value = "".join(map(str, init_state))
        x_value = int(x_value, 2) / (2**n)

        # Define Signal Operator
        sig_op = lambda x: np.array(
                [[x, np.sqrt(1 - x**2)],
                [np.sqrt(1 - x**2), -x]])
        
        # Define Angle Operator
        qsp_op = lambda phi: np.array(
            [[np.exp(1j * phi), 0.],
                [0., np.exp(-1j * phi)]])
        
        angle_matrices = []
        for phi in angle_list:
            angle_matrices.append(qsp_op(phi))
            

        R = sig_op(x_value)
        U = angle_matrices[0]
        for angle_matrix in angle_matrices[1:]:
            U = U @ R @ angle_matrix
        
        res = U[0, 0] # A complex number
        im_res = res.imag
        re_res = res.real
        print("===="*10)
        print(f"Real part of encoded function @ x = {x_value}:\n\t {re_res}\n")
        print(f"Imaginary part of encoded function @ x = {x_value}: \n\t{im_res}\n")
        print(f"Target function value at x = {x_value}: \n\t{func(x_value)}")
        print("===="*10)

    return raw_sv, dirac_sv

def FullSVFromStatePrepCircuit(angle_list: list, num_ws_qbs: int) -> tuple[np.array, str]:
    """
    Generates the full state vector from the QSP state preparation circuit.
    Args:
        angle_list: A list of angles used in the QSP state preparation.
        num_ws_qbs: The number of workspace qubits used in the circuit.
    Returns:
        A tuple containing the raw state vector and its Dirac notation.
    """
    # setup circuit 
    circuit = StatePrep(angle_list, num_ws_qbs)

    raw_sv = cirq.final_state_vector(circuit)
    dirac_sv = cirq.dirac_notation(raw_sv)

    return raw_sv, dirac_sv
    