# AngleAnalysis.py
# Gabriel Waite

# !pip install pyqsp
import pyqsp
from pyqsp import angle_sequence
import matplotlib.pyplot as plt
import numpy as np
import math
import CoefficientFinder as cf

# --- -- -- - - - Degree Estimation Functions - - -- -- --- #

def AnalyticalDegreeFromError(sigma: float, error: float) -> int:
    """
    Computes a lower bound on 2*degree based on the error tolerance and sigma.
    Args:
      sigma (float): The standard deviation of the Gaussian.
      error (float): The desired error tolerance.
    Returns:
      int: The estimated 2*degree.

    Note:
      Output is rounded up to the nearest even integer.
    """
    return math.ceil((1 / (1-np.sin(1))) * ((np.pi**2 / (8 * sigma**2)) + np.log(1/error))  / 2) * 2 

def NumericalDegreeFromError(sigma: float, error: float, n: int, initial_degree = None, verbose: bool = False, plot: bool = False) -> int:
    """
    Computes an estimate on 2*degree based on the error tolerance and sigma.
    Args:
      sigma (float): The standard deviation of the Gaussian.
      error (float): The desired error tolerance.
      n (int): The number of qubits
      initial_degree : A guess for the degree of a polynomial (Even number, **not** the index)
      verbose (bool): If True, prints descriptive messages about the degree bounds and adjustments.
      plot (bool): If True, plots the approximated function against the target function and the absolute difference.
    Returns:
      int: The estimated 2*degree.
    Note:
      For very small error it is good to try guess a smaller initial_degree.
    """
    if initial_degree is None:
      initial_degree = AnalyticalDegreeFromError(sigma, error)

    print(f"Initial degree guess: {initial_degree}\n") if verbose else None
    max_val = np.sin(1 - (1/(2**n)))
    x_eval = np.linspace(0.5,max_val, 2**n)
    targ_func = lambda x: np.exp( - (1/(2 * sigma**2)) * (np.arcsin(x))**2)

    d = initial_degree
    delta = 0
    final_poly = None
    
    while True:
      init = cf.TargetSeriesCoefficients(sigma=sigma)
      coeff_list = init.ListTargetTaylorCoeff(d//2)
      poly = init.CoeffListToPolynomial(coeff_list, "TAYLOR")
      delta = np.max(np.abs(poly(x_eval) - targ_func(x_eval)))

      print("-----"* 10) if verbose else None
      print(f"degree={d} gives delta={delta}") if verbose else None
      print("-----"* 10) if verbose else None

      print(f"Value of x @ degree {d} giving largest delta: {x_eval[np.argmax(np.abs(poly(x_eval) - targ_func(x_eval)))]}\n") if verbose else None

      if delta <= error:
        # Try a smaller degree
          final_poly = poly
          d -= 2
          if d < 0:
            break 
      else:
        final_deg = d + 2
        init = cf.TargetSeriesCoefficients(sigma=sigma)
        coeff_list = init.ListTargetTaylorCoeff(final_deg // 2)
        final_poly = init.CoeffListToPolynomial(coeff_list, "TAYLOR")

        if plot == True:
          _, axs = plt.subplots(1, 2, figsize=(20, 10))
          x = np.linspace(0, max_val, 100)
          # Plot 1: Target Function and Approximated Function
          axs[0].plot(x, targ_func(x), label=f'Target Function @ sigma={sigma}', color='blue')
          axs[0].plot(x, final_poly(x), label=f'Approximated Function @ degree={final_deg}', color='orange')
          axs[0].set_title(f'Approximation of Target Function with Degree {final_deg}')
          axs[0].set_xlabel('x')
          axs[0].set_ylabel('Function Value')
          axs[0].legend()
          axs[0].grid() 

          # Plot 2: Absolute Difference
          axs[1].semilogy(x, np.abs(poly(x) - targ_func(x)), label='Absolute Difference', color='red')
          axs[1].axhline(y=error, color='green', linestyle='--', label='Error Tolerance')
          axs[1].axvline(x=np.sin(1 - (1/(2**n))), color='purple', linestyle='--', label='Max of domain')
          axs[1].set_title('Absolute Difference between Target and Approximated Function')
          axs[1].set_xlabel('x')
          axs[1].set_ylabel('Absolute Difference')
          axs[1].legend()
          axs[1].grid()

          plt.show()
        # If previous is too small, increment degree
        print("-----"* 10) if verbose else None
        print(f"Final degree: {final_deg} | {d+2}") if verbose else None
        print("-----"* 10) if verbose else None
        return final_deg
      

# --- -- -- - - - Quantum Signal Processing Functions - - -- -- --- #

def angle_shift(angle_list: list) -> list:
    """
    Shifts the angles in the angle list by pi/4 or pi/2.
    Args:
        angle_list (list): List of angles to be shifted from pyqsp.
    Returns:
        list: List of shifted angles.
    Note:
        This is necessary because the angles returned by pyqsp are w.r.t W(x) operator, but we use R(x) operator.
    """
    shifted_angle_set = []
    for phi in range(len(angle_list)):
        if phi == 0 or phi == len(angle_list) - 1:
            shifted_angle_set.append(angle_list[phi] - np.pi/4)
        else:
            shifted_angle_set.append(angle_list[phi] - np.pi/2)

    return shifted_angle_set

def QSPTest(angle_list: list, x_data: list) -> tuple:
    """
    Computes the output of a Quantum Signal Processing (QSP) operation for given angles and input data.
    Args:
        angle_list (list): List of angles for the QSP operation.
        x_data (list): List of input data points to apply the QSP operation on.
    Returns:
        tuple: A tuple containing:
            - y_data (list): The output data after applying the QSP operation.
            - imag_data (list): The imaginary parts of the output data.
            - real_data (list): The real parts of the output data.
    Note:
        ** Important: The signal operator we use is not the standard W(x) operator, but one we call R(x) **
            W(x) = [[x, 1j * sqrt(1 - x^2)], [1j * sqrt(1 - x^2), x]]
            R(x) = [[x, sqrt(1 - x^2)], [sqrt(1 - x^2), -x]]
        It follows that:
            R(x) = -1j * exp(1j * pi/4) * W(x) * exp(1j * pi/4)
        ** If output function is not as expected, try shifting the angles **
    """
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
        
    y_data = []
    for x in x_data:
        R = (1j) * sig_op(x)
        U = angle_matrices[0]
        for angle_matrix in angle_matrices[1:]:
            U = U @ R @ angle_matrix
        y_data.append(U[0, 0]) # Take the top left element as the output

    return y_data, [np.imag(y) for y in y_data], [np.real(y) for y in y_data]

def GetQSPAngleList(sigma: float, max_degree: int) -> list:
    """
    Computes the QSP angle list for a given sigma and maximum degree.
    Args:
        sigma (float): The standard deviation of the Gaussian.
        max_degree (int): The maximum degree for the polynomial.
    Returns:
        list: A list of angles for the QSP approximation.
    """
    if max_degree < 0 or max_degree % 2 != 0:
        raise ValueError("max_degree must be a non-negative even integer.")

    X = cf.TargetSeriesCoefficients(sigma) # Initialise the instance
    coeff_list = X.ListTargetChebyCoeff(int(max_degree//2)) # Recall that ListTargetChebyCoeff takes a even number corresponding to index of coefficient.
    poly_data = X.CoeffListToPolynomial(coeff_list, "CHEBYSHEV")

    return angle_shift(angle_sequence.QuantumSignalProcessingPhases(poly_data, method='sym_qsp', chebyshev_basis=True)[0])

def GetQSPAngleListAdv(sigma: float, init_trunc_degree: int, error: float, plot: bool=False, verbose: bool=True) -> tuple:
    """
    Computes the QSP angle list for a given sigma and error tolerance.
    Capable of outputting a plot of the approximated function against the target function and the absolute difference.

    Args:
        sigma (float): The standard deviation of the Gaussian.
        init_trunc_degree (int): The initial maximum degree for the polynomial.
        error (float): The desired error tolerance.
        plot (bool): If True, plots the approximated function against the target function and the absolute difference.
        verbose (bool): If True, prints descriptive messages about the degree bounds and adjustments.
    Returns:
        tuple: A tuple containing:
            - angle_list (list): The list of angles for the QSP approximation.
            - working_degree (int): The working degree used for the approximation.
            - numerical_degree (int): The numerical degree based on the error tolerance.
            - naive_degree (int): The naive degree based on the error tolerance.
    Note:
        The function first checks if the init_trunc_degree is a good guess by comparing it to the naive bound and the numerical bound.
        If the init_trunc_degree is smaller than the naive bound, it will print a warning and suggest a new starting bound based on the numerical bound.
        The function then computes the QSP angle list using the working degree, which is either the init_trunc_degree or the numerical bound.
        If plot is True, it will plot the approximated function against the target function and the absolute difference.
        **We require a degree as input and not an index, so the input should be an even integer.**
        We shift the angles by pi/4 or pi/2 to account for the difference between the W(x) and R(x) operators.
    """

    if init_trunc_degree < 0 or init_trunc_degree % 2 != 0:
        raise ValueError("init_trunc_degree must be a non-negative even integer.")
    # Initialise degree bounds
    # number of qubits is set to 10, but can be changed.
    naive_degree, numerical_degree, working_degree = AnalyticalDegreeFromError(sigma, error), \
                                                     NumericalDegreeFromError(sigma, error, 10, initial_degree=init_trunc_degree), \
                                                     init_trunc_degree

    # Compare init_trunc_degree to AnalyticalDegreeFromError function.
    if (init_trunc_degree < naive_degree) is True:
        print(f"Your initial degree guess of {init_trunc_degree} is smaller than the naive bound of {naive_degree}.\n") if verbose else None
        if (init_trunc_degree < numerical_degree) is True:
            print(f"Your initial degree guess of {init_trunc_degree} is also smaller than the numerical bound {numerical_degree}.\n \
                        (!) We will use the numerical bound *or* you can re-run the function with a better starting degree.(!) \n \
                        (1) We suggest a new starting bound of {numerical_degree + 4}. (!)") if verbose else None
            working_degree = numerical_degree
            print("-----"*10) if verbose else None
            print(f"Your initial degree guess of {init_trunc_degree} is reduced to {working_degree}(!)") if verbose else None
            print("-----"*10) if verbose else None
            print("\n") if verbose else None
        else:
            print(f"Your initial degree guess of {init_trunc_degree} is larger than the numerical bound of {numerical_degree}.\n (!)We will use the numerical bound.(!)\n") if verbose else None
            working_degree = numerical_degree
            print("-----"*10) if verbose else None
            print(f"Your initial degree guess of {init_trunc_degree} is reduced to {working_degree}(!)") if verbose else None
            print("-----"*10) if verbose else None
            print("\n") if verbose else None
    else:
        print(f"Your initial degree guess {init_trunc_degree} is larger than the naive bound of {naive_degree}, so we start from the latter and try get a better bound.\n") if verbose else None
        working_degree = NumericalDegreeFromError(sigma, error, 10, initial_degree = naive_degree)
        print("-----"*10) if verbose else None
        print(f"Your initial degree guess {init_trunc_degree} is reduced to {working_degree}(!)") if verbose else None
        print("-----"*10) if verbose else None
        print("\n") if verbose else None
  
    print("-----"*10) if verbose else None
    print(f"> The working degree is now {working_degree} <") if verbose else None
    print("-----"*10) if verbose else None
    print("\n") if verbose else None

    # Add code to allow user to overrule
    user_input = input(f"Do you want to use a different working degree? (y/n): ")
    if user_input.lower() == 'y':
        while True:
            try:
                new_degree = int(input("Enter the desired even, non-negative degree: "))
                if new_degree >= 0 and new_degree % 2 == 0:
                    working_degree = new_degree
                    print("-----"*10) if verbose else None
                    print(f"> Using user-specified degree: {working_degree} <") if verbose else None
                    print("-----"*10) if verbose else None
                    break
                else:
                    print("Invalid input. Please enter an even, non-negative integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
    else:
        print("Using the calculated working degree.") if verbose else None

    # Given the working_degree we can call GetQSPAngleList.
    # Recall that the working_degree is twice the index we need to call.
    print(f"This is the type of the working degree divided by 2: {type(working_degree//2)}") 
    angle_list = GetQSPAngleList(sigma, working_degree)

    if plot == True:
        y_plot_data = QSPTest(angle_list, np.linspace(0, 1, 100))[1]
        y_targ_func_data = [np.exp(-(1/(2 * sigma**2)) * (np.arcsin(x))**2) for x in np.linspace(0, 1, 100)]

        _, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Plot 1: Approximate and Target functions
        axs[0].plot(np.linspace(0, 1, 100), y_plot_data, color='red', label='Approximated function')
        axs[0].plot(np.linspace(0, 1, 100), y_targ_func_data, color='black', label='Target function')
        axs[0].axvline(x=np.sin(1), color='g', linestyle='--', label='x = sin(1)')
        axs[0].set_xlabel('X-axis')
        axs[0].set_ylabel('Y-axis')
        axs[0].set_title('Action of the approximation polynomial on the Wx operator')
        axs[0].grid(True)
        axs[0].legend()

        # Plot 2: Absolute difference
        axs[1].semilogy(np.linspace(0, 1, 100), np.abs(np.array(y_plot_data) - np.array(y_targ_func_data)), color='blue', label='Absolute difference')
        axs[1].axhline(y=error, color='r', linestyle='--', label=f'Error = {error}')
        axs[1].axvline(x=np.sin(1), color='g', linestyle='--', label='x = sin(1)')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')
        axs[1].set_title('Absolute difference between the approximated and target functions')
        axs[1].grid(True)
        axs[1].legend()
            
        plt.show() if plot else None

    return angle_shift(angle_list), working_degree, numerical_degree, naive_degree

def SaveQSPAngleListToFile(sigma: float, init_degree: int, error: float, filename: str):
    if filename is None or not isinstance(filename, str):
        raise ValueError("Please provide a file name ending with '.txt'.")
    if not filename.endswith('.txt'):
        raise ValueError("Filename must end with .txt")
    
    result = GetQSPAngleListAdv(sigma, init_degree, error, plot=False)
    angle_list = result[0]
    working_degree = result[1]
    with open(filename, 'w') as f:
        f.write(f"% File: {filename}\n")
        f.write(f"% Input Data = [{sigma},{init_degree},{working_degree},{error}]\n")
        f.write("QSP angles form PYQSP\n")
        f.write("————————————————————\n")
        f.write(f"sigma = {sigma}\n")
        f.write(f"Initial Degree = {init_degree}\n")
        f.write(f"Working Degree = {working_degree}\n")
        f.write(f"Error = {error}\n")
        f.write("————————————————————\n\n")
        f.write(str(angle_list))
        f.write("\n")