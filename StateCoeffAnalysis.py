# StateCoeffAnalysis.py
# Gabriel Waite

import cirq
import numpy as np
import matplotlib.pyplot as plt
import QSPStatePrep as qsp

def SignatureVectorElements(vector: np.array, signature: list[int], len_rem_bits: int) -> list[tuple[float, float]]:
    """
    Extracts the coefficients of a state vector based on a given signature and remaining bits.
    Args:
        vector: The state vector from which to extract coefficients.
        signature: A list of bits representing the signature.
        len_rem_bits: The number of remaining bits after the signature.
    Returns:
        A list of tuples containing the real and imaginary parts of the coefficients.
    Raises:
        ValueError: If the number of indices does not match the vector dimension.
    Note:
        The signature is used to define the indices of the coefficients in the vector.
        The remaining bits are appended to the signature to form the full index.
    """
    # We want to use the signature to define the indices.
    if 2**(len(signature) + len_rem_bits) != len(vector):
        raise ValueError("Check the number of indices match the vector dimension!")
    indices = []
    sig = "".join(map(str,signature))
    rem_bit_ind = [f'{i:0{len_rem_bits}b}'[::-1] for i in range(2**len_rem_bits)]
    # We reverse them due to how the statevector from cirq is defined

    indices = [(i, sig + v) for i, v in enumerate(rem_bit_ind)]

    coefficients = [(float(vector[int(i[1],2)].real),float(vector[int(i[1],2)].imag)) for i in indices]

    return coefficients

def PlotSignatureVectorElements(coefficients: list[tuple], num_qs_qbs: int, sigma: float = 1.0) -> None:
    """
    Plots the real and imaginary parts of the coefficients from the signature vector against the target function.
    Args:
        coefficients: A list of tuples containing the real and imaginary parts of the coefficients.
    Returns:
        None
    """
    n,l = num_qs_qbs, len(coefficients)
    x = np.arange(l)
    re_coeff, im_coeff = [i[0] for i in coefficients], [i[1] for i in coefficients]
    r = max(im_coeff)
    
    func = lambda x: r * np.exp(- ((x/2**n)**2) / (2 * sigma**2))

    plt.scatter(x, re_coeff, label='Real')
    plt.scatter(x, im_coeff, label='Imaginary')
    plt.plot(x, re_coeff, label='Real', alpha=0.5)
    plt.plot(x, im_coeff, label='Imaginary', alpha=0.5)
    plt.plot(np.linspace(0,2**n - 1,50), func(np.linspace(0,2**n - 1,50)), label='Target')
    plt.xlabel('Index')
    plt.ylabel('Coefficient')
    plt.title('Signature Vector Coefficients')
    plt.legend()
    plt.show()

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
    circuit = qsp.StatePrep(angle_list, num_ws_qbs)

    raw_sv = cirq.final_state_vector(circuit)
    dirac_sv = cirq.dirac_notation(raw_sv)

    return raw_sv, dirac_sv

if __name__ == "__main__":
    # Example usage
    sigma, n = 0.5, 3
    test_angle_list = [np.float64(-0.7854005535036002),
    np.float64(-1.5706035611195723),
    np.float64(-1.577339892447164),
    np.float64(-1.4886573704222252),
    np.float64(-1.9534042292441365),
    np.float64(-0.9429716850622075),
    np.float64(-1.9534042292441365),
    np.float64(-1.4886573704222252),
    np.float64(-1.577339892447164),
    np.float64(-1.5706035611195723),
    np.float64(-0.7854005535036002)]

    # Seems that we want to post select the signature bits '10'
    arr = qsp.FullSVFromStatePrepCircuit(test_angle_list,n)

    vec = arr[0]
    PlotSignatureVectorElements(SignatureVectorElements(vec, [1,0], n), n, sigma)