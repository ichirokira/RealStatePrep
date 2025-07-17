# QSPStatePrep.py
# Gabriel Waite

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
        return cirq.CircuitDiagramInfo(wire_symbols=("a","a"))
    
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
  n = num_ws_qbs

  # setup circuit qubits
  qsp_anc, be_anc = cirq.NamedQubit('qsp_anc'), cirq.NamedQubit('rbe_anc')
  workspace = [cirq.NamedQubit(f'x_{i}') for i in range(1, n + 1)]

  be_qbs = [be_anc] + workspace

  # Initialise circuit
  circuit = cirq.Circuit()
  circuit.append(cirq.H.on_each(workspace))

  BE_gate = ControlledRySequenceGate(num_workspace_qubits=n)

  for i in range(len(angle_list)-1,0,-1):
    print(angle_list[i])
    circuit.append(CRGate(phi=angle_list[i])(be_anc, qsp_anc))
    circuit.append(BE_gate(*be_qbs))
  # add zeroth angle
  circuit.append(CRGate(phi=angle_list[0])(be_anc, qsp_anc))

  return circuit