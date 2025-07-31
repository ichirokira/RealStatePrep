import sys
sys.path.append("../")
import cirq
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from ops.qsp_state_prep import QSPStatePrepAA


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--system_nqubits', default=4)
    parser.add_argument('-err', '--error', default=1e-6)
    parser.add_argument('-s', '--sigma', default=0.25)
    parser.add_argument('-f', '--figure', default='../results/qsp_stateprep_output.png',
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

if __name__ == '__main__':

    config = get_argparse()
    system_nqubits = config.system_nqubits
    y = generate_coeff(system_nqubits, sigma = config.sigma)
    state_prep =  QSPStatePrepAA(system_nqubits=system_nqubits, sigma = config.sigma, error = config.error)
    state_prep.construct_circuit(num_iteration = math.ceil((np.pi/4)*np.sqrt(2**system_nqubits)/np.linalg.norm(y)))

    amplitudes = state_prep.get_output()
    
    plt.plot(np.arange(0,2**system_nqubits), y/np.linalg.norm(y), 'o-', color='blue', label = 'Target')
    plt.plot(np.arange(0,2**system_nqubits), amplitudes, 'o-', color='orange', label='Output')
    plt.legend()
    plt.savefig(config.figure)