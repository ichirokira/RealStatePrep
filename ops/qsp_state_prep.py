import sys
sys.path.append("../")
import cirq
import numpy as np
import math
import argparse
import utils.qsp.CoefficientFinder as cf
import utils.qsp.AngleAnalysis as aa
import utils.qsp.QSPStatePrep as qsp
import utils.qsp.StateCoeffAnalysis as sca
import matplotlib.pyplot as plt

class QSPStatePrepAA:

    def __init__(self, 
                 system_nqubits: int = 2,
                 sigma: float = 0.25,
                 error: float = 1e-6):
        
        self.system_nqubits = system_nqubits
        self.sigma = sigma
        self.error = error
        # Generate coefficients
        X = cf.TargetSeriesCoefficients(sigma = self.sigma)

        # # Estimate the degree
        improved_degree =  aa.NumericalDegreeFromError(self.sigma, self.error, self.system_nqubits)

        # # Get the QSP angles
        self.qsp_angles = aa.GetQSPAngleListAdv(self.sigma, improved_degree, self.error, plot = False, verbose = False)

        # Define Qubits
        self.qsp_anc, self.be_anc = cirq.NamedQubit('qsp_anc'), cirq.NamedQubit('rbe_anc')
        self.workspace = [cirq.NamedQubit(f'x_{i}') for i in range(1, self.system_nqubits + 1)]

    def qsp_circuit(self,):

        angle_list = self.qsp_angles[0]
        be_qbs = [self.be_anc] + self.workspace

        # Initialise circuit
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on(self.qsp_anc))
        circuit.append(cirq.H.on_each(self.workspace))

        BE_gate = qsp.ControlledRySequenceGate(num_workspace_qubits=self.system_nqubits)

        for i in range(len(angle_list)-1,0,-1):
            print(angle_list[i])
            circuit.append(qsp.CRGate(phi=angle_list[i])(self.be_anc, self.qsp_anc))
            circuit.append(BE_gate(*be_qbs))
        # add zeroth angle
        circuit.append(qsp.CRGate(phi=angle_list[0])(self.be_anc, self.qsp_anc))

        circuit.append(cirq.H.on(self.qsp_anc))
       
        return circuit

    
    def phase_oracle(self, ) -> cirq.Circuit:
      """
      Negate the amplitude with |1>_qsp_acn |0>_be_anc
      """
      circ = cirq.Circuit()
      circ.append(cirq.X(self.be_anc))
      circ.append(cirq.H(self.be_anc)) 
      circ.append(cirq.XPowGate().controlled(1, control_values=[1]).on(self.qsp_anc, self.be_anc))
      circ.append(cirq.H(self.be_anc)) 
      circ.append(cirq.X(self.be_anc))
      return circ

    def zero_reflection(self, qubits: list) -> cirq.Circuit:
      """
      Reflect zero state.
      Implement I - 2|0><0| over all qubits
      """
      circ = cirq.Circuit()
      for i in range(len(qubits)):
        circ.append(cirq.X(qubits[i]))

      circ.append(cirq.H(qubits[-1])) # on be_anc
      circ.append(cirq.XPowGate().controlled(self.system_nqubits+1).on(*qubits))
      circ.append(cirq.H(qubits[-1])) # on be_anc

      for i in range(len(qubits)):
        circ.append(cirq.X(qubits[i]))

      return circ

    def amplitude_amplification(self, num_iteration: int) -> cirq.Circuit:
        circ = cirq.Circuit()

        phase_oracle = self.phase_oracle()
        good_state_preparation = self.qsp_circuit()
        reflection = self.zero_reflection(self.workspace+[self.qsp_anc]+[self.be_anc])

        circ.append(good_state_preparation, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        for i in range(num_iteration):
            circ.append(phase_oracle, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            circ.append(cirq.inverse(good_state_preparation), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            circ.append(reflection, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            circ.append(good_state_preparation, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        return circ
    def construct_circuit(self,num_iteration: int) -> None:
        self.circuit = self.amplitude_amplification(num_iteration=num_iteration) 

    def get_output(self) -> list:
        # measurement
        # raw_sv = cirq.final_state_vector(self.circuit)
        # coeff = sca.SignatureVectorElements(raw_sv, [1,0], self.system_nqubits)
        # sca.PlotSignatureVectorElements(coeff, self.system_nqubits, self.sigma)
        self.circuit.append(cirq.measure(*(self.workspace[::-1]+[self.qsp_anc]+[self.be_anc]), key="result"))
        s = cirq.Simulator()
        samples = s.run(self.circuit, repetitions=1000)
        samples = cirq.ResultDict(records={'result': samples.records['result']})
        results = samples.histogram(key="result")
        # print(results)
        care_results = []

        for i in range(2**self.system_nqubits):
            binarized_i = format(i, "b").zfill(self.system_nqubits)
            
            #binarized_i = binarized_i[::-1]
            care_position = int(binarized_i+"10", 2)
            # care_position = i
            care_results.append(results[care_position])

        amplitude = np.sqrt(np.array(care_results)/(sum(care_results)))

        return amplitude