# QSPStatePrep.py

import cirq

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
    
def StatePreparation(num_ws_qbs: int, angle_list: list) -> cirq.Circuit:
    """
    Implements the state preparation circuit using QSP with controlled Ry rotations.
    Args:
        sigma (float): The parameter for the target function.
        num_ws_qbs (int): The number of workspace qubits.
        angle_list (list): A list of angles for the QSP rotations.
    Returns:
        cirq.Circuit: The circuit implementing the state preparation.
    """
    n = num_ws_qbs # Number of workspace qubits

    # Setup circuit qubits
    ancilla = [cirq.NamedQubit(f'{desc}_anc') for desc in ['qsp','rbe']] # quantum signal processing ancilla and rotation block encoding ancilla
    workspace = [cirq.NamedQubit(f'x_{i}') for i in range(1,n+1)]

    # Initialise the circuit
    circuit = cirq.Circuit()
    # Hadamard on first ancilla and all workspace
    circuit.append(cirq.H(ancilla[0]))
    circuit.append(cirq.H.on_each(workspace))

    # cirq.rz(rads) is equivalent to 
    #  exp(-i Z rads / 2) = cos(rads/2) I - i sin(rads/2) Z
    #                     | 
    #                     = [[exp(-i rads/2), 0], [0, exp(i rads/2)]]
    # Define QSP operator
    #   qsp_op = [[np.exp(1j * phi), 0.], [0., np.exp(-1j * phi)]]

    # Apply first QSP operator on first ancilla
    phi_0 = angle_list[0]
    circuit.append(cirq.rz(-2 * phi_0).on(ancilla[0]))


    my_custom_gate = ControlledRySequenceGate(num_workspace_qubits=n)
    be_qbs = [ancilla[1]] + workspace
    # Apply RZ rotations for phi in the list from 1 onwards.
    # Between each rotation, do a controlled unitary BE.
    for phi in angle_list[1:]:
        circuit.append(my_custom_gate.on(*be_qbs).controlled_by(ancilla[0]))
        circuit.append(cirq.rz(-2 * phi).on(ancilla[0]))

    return circuit