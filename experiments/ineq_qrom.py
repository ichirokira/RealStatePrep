import matplotlib.pyplot as plt
import cirq
import math
import numpy as np
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran._infra.gate_with_registers import get_named_qubits
from qualtran.cirq_interop import _bloq_to_cirq
from cirq.contrib.svg import SVGCircuit
import sys
sys.path.append("../")
from ops.ineq_state_prep import StatePrepIneq
import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--system_nqubits', default=2)
    parser.add_argument('-m', '--data_nqubits', default=2)
    parser.add_argument('-s', '--sigma', default=1.0)
    parser.add_argument('-f', '--figure', default='../results/ineq_qrom_output.png',
                        help='Output file for the visualization')
    return parser.parse_args()
def generate_coeff(n: int, sigma: float) -> np.ndarray:
  """
  Compute a vector of 2**n coefficients of gaussian function
  """

  N = 2**n
  x = np.arange(N)
  x = (x) / (N)
  coeffs = np.exp(-(x**2/(2*sigma**2)))
  return coeffs

def blackbox(selection_register, target_register, qrom):
  qrom_cirq = qrom.as_composite_bloq().to_cirq_circuit(
    cirq_quregs= {'selection': selection_register, 'target0_':target_register}
  )
  return qrom_cirq

if __name__ == '__main__':

    config = get_argparse()
    system_nqubits = config.system_nqubits
    data_nqubits = config.data_nqubits
    y = generate_coeff(system_nqubits, sigma = config.sigma)
    #y_qrom = np.arange(2**system_nqubits)[::-1]
    y_qrom = np.round(y*(2**(data_nqubits)-1))
    
    qrom = QROM([y_qrom], selection_bitsizes=(system_nqubits,), target_bitsizes=(data_nqubits,))
    state_prep = StatePrepIneq(num_out_qubits=system_nqubits, num_data_qubits=data_nqubits, qrom=qrom)
    state_prep.construct_circuit(num_iteration = math.ceil((np.pi/4)*np.sqrt(2**system_nqubits)/np.linalg.norm(y)))

    amplitudes = state_prep.get_output()
    plt.plot(np.arange(0,2**system_nqubits), y_qrom/np.linalg.norm(y_qrom), 'o-', color='blue', label = 'Target')
    plt.plot(np.arange(0,2**system_nqubits), amplitudes, 'o-', color='orange', label='Output')
    plt.legend()
    plt.savefig(config.figure)