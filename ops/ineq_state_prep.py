import sys
sys.path.append("../")
import cirq
import numpy as np
from utils.comparator import Comparator


def blackbox(selection_register, target_register, qrom):
  "Define Blacbox as QROM"
  qrom_cirq = qrom.as_composite_bloq().to_cirq_circuit(
    cirq_quregs= {'selection': selection_register, 'target0_':target_register}
  )
  return qrom_cirq

class StatePrepIneq:
  """
  Implement inequality test based method in https://arxiv.org/abs/1807.03206 with regular ampltitude amplification technqiues

  Attributes:
    -----------
    num_out_qubits : int
        Number of qubits for output register

    num_data_qubits : int
        Number of qubits to estimate the output data from the oracle

    black_box: cirq.Circuit
        The oracle to generate the output data.

    input_data : List[float]
        (Optional) List of amplitudes when black_box is None

  Methods:
    --------
    good_state_preparation() -> cirq.Circuit
        Generates 'good state' cirq.Circuit.

    amplitude_amplification(num_iteration) -> cirq.Circuit
        Generates the amplitude amplification circuit based on the 'good' circuit.

    construct_circuit() -> None
        Combine good_state_preparation and amplitude_amplification circuit

    get_output() -> List[float]
        Simulate results and get ampltiudes of the output state
  """
  def __init__(self, num_out_qubits: int, num_data_qubits: int, input_data: list =None, qrom: cirq.Circuit = None)-> None:
    """
    :param num_out_qubits: number of qubits for output register
    :param num_data_qubits: number of qubits to estimate the output data from the oracle
    :param input_data: if 'black_box' is None, user can direct input the list of amplitudes as 'input_data'
    :param qrom: the oracle to generate the output data.
    """
    self.num_out_qubits = num_out_qubits
    self.num_data_qubits = num_data_qubits
    self.input_data = input_data

    self.out = [cirq.NamedQubit('out' + str(i)) for i in range(num_out_qubits)]
    self.data = [cirq.NamedQubit('data' + str(i)) for i in range(num_data_qubits)]
    self.ref = [cirq.NamedQubit('ref' + str(i)) for i in range(num_data_qubits)]
    self.flag = cirq.NamedQubit('flag')
    
    self.qrom = qrom

  def good_state_preparation(self,) -> cirq.Circuit:
    """
    Generate the good state at |0>_ref|0>-flag.
    Implement Eq.(7) in the paper
    """
    circuit = cirq.Circuit()

    # intializer
    for i in range(self.num_out_qubits):
      circuit.append(cirq.H(self.out[i]))
   
    circuit.append(blackbox(self.out, self.data, qrom=self.qrom), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    # initialize ref
    for i in range(self.num_data_qubits):
      circuit.append(cirq.H(self.ref[i]))

    # compare ref to data
    comparator = Comparator(self.ref, self.data)
    compare_circ = comparator.construct_circuit()

    circuit.append(compare_circ, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    circuit.append(cirq.X(comparator.anc1))
    circuit.append(cirq.CNOT(comparator.anc1, self.flag))
    circuit.append(cirq.X(comparator.anc1))

    # uncompute comparator
    inv_compare = cirq.inverse(compare_circ)
    circuit.append(inv_compare, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    # unprepare superposition from ref
    for i in range(self.num_data_qubits):
      circuit.append(cirq.H(self.ref[i]))

    return circuit

  def amplitude_amplification(self, num_iteration: int) -> cirq.Circuit:

    #define components
    def phase_oracle() -> cirq.Circuit:
      """
      Negate the amplitude with |0>_ref |0>_flag
      """
      circ = cirq.Circuit()
      circ.append(cirq.X(self.flag))
      circ.append(cirq.H(self.flag)) # on flag
      circ.append(cirq.XPowGate().controlled(self.num_data_qubits, control_values=[0]*self.num_data_qubits).on(*self.ref, self.flag))
      circ.append(cirq.H(self.flag)) # on flag
      circ.append(cirq.X(self.flag))
      return circ

    def zero_reflection(qubits: list) -> cirq.Circuit:
      """
      Reflect zero state.
      Implement I - 2|0><0| over all qubits
      """
      circ = cirq.Circuit()
      for i in range(len(qubits)):
        circ.append(cirq.X(qubits[i]))

      circ.append(cirq.H(qubits[-1])) # on flag
      circ.append(cirq.XPowGate().controlled(self.num_out_qubits+2*self.num_data_qubits).on(*qubits))
      circ.append(cirq.H(qubits[-1])) # on flag

      for i in range(len(qubits)):
        circ.append(cirq.X(qubits[i]))

      return circ

    circ = cirq.Circuit()
    oracle = phase_oracle()
    good_state_preparation = self.good_state_preparation()
    reflection = zero_reflection(self.out+self.data+self.ref+[self.flag])


    circ.append(good_state_preparation, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    for i in range(num_iteration):
      circ.append(oracle, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
      circ.append(cirq.inverse(good_state_preparation), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
      circ.append(reflection, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
      circ.append(good_state_preparation, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circ

  def construct_circuit(self,num_iteration: int) -> None:
    self.output_circuit = self.amplitude_amplification(num_iteration=num_iteration) # math.floor(np.sqrt(2**self.num_data_qubits))
    # uncompute data
    self.output_circuit.append(cirq.inverse(blackbox(self.out, self.data, qrom=self.qrom)), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

  def get_output(self) -> list:
    # measurement
    self.output_circuit.append(cirq.measure(*(self.out[::-1]+self.ref[::-1]+[self.flag]), key="result"))
    s = cirq.Simulator()
    samples = s.run(self.output_circuit, repetitions=1000)
    samples = cirq.ResultDict(records={'result': samples.records['result']})
    results = samples.histogram(key="result")
    care_results = []
    for i in range(2**self.num_out_qubits):
      binarized_i = format(i, "b").zfill(self.num_out_qubits)
      
      # binarized_i = binarized_i[::-1]
      care_position = int(binarized_i+"0"*self.num_data_qubits+"0", 2)
      care_results.append(results[care_position])

    amplitude = np.sqrt(np.array(care_results)/(sum(care_results)))

    return amplitude