# CoefficientFinder.py

import numpy as np 
import math 
import json
from numpy.polynomial import Polynomial, chebyshev

class TargetSeriesCoefficients:
    """
    A class to compute coefficients for the series expansion of the target function:
        F(x) = Exp[ - 1/(2*sigma^2) * arcsin(x)^2 ].

    **Attributes:**
    > "sigma" : A positive float representing the standard deviation of the Gaussian function.
    > "Coefficient" : refers to the coefficients in a series expansion, which can be either Taylor or Chebyshev coefficients.
    > "(Coefficient) Index" : refers to the i associated with the coefficient c_i in the series expansion.
    > "Degree" : refers to the degree of the polynomial or Chebyshev polynomial, which is 2 times the index (i.e., degree = 2 * index).

    The class supports two primary bases for the Taylor series expansions about x=0:

    1.  **Standard Polynomial Taylor Series (P_d(x)):**
        Represented as a truncated polynomial:
        P_d(x) = 1 + sum_{n=1}^{d} c_n * x^(2n)
        Where 'd' is the maximum coefficient index (corresponding to a polynomial of degree 2d). 
        The coefficients 'c_n' are computed using Bell polynomials.
        (We refer to the Standard Polynomial series as "Taylor" series in the context of this class).

    2.  **Chebyshev Series of the First Kind (T_n(x)):**
        Represented as:
        Q_d(x) = sum_{n=0}^{d} l_n * T_{2n}(x)
        Where 'd' is the maximum coefficient index (corresponding to the Chebyshev of degree 2d) and 'l_n' are the Chebyshev coefficients.
        Note that only even-indexed Chebyshev polynomials (T_0, T_2, T_4, ...) are used due to the even nature of the target function's Taylor expansion.
        The conversion from Taylor to Chebyshev coefficients involves pre-computed Chebyshev coefficient arrays loaded from 'cheby_coeffs.json'.

    **Key Assumptions & Conventions:**
    * **Input Range for x:** While not explicitly enforced by methods, the arcsin(x) function
        is real-valued for x in [-1, 1]. The series expansions are generally valid within this range.
    * **Coefficient Ordering:** All lists/arrays of coefficients (e.g., `c_n`, `l_n`) are ordered by increasing index. 
        For a list representing coefficients up to index 'I', the element at index `i` corresponds to the coefficient of the either x^(2i) for Standard Polynomial series or T_{2i}(x) for Chebyshev series.
    * **Implied Zeros for Odd Powers:** For Standard Polynomial and Chebyshev expansions where only even powers/polynomials appear (e.g., x^0, x^2, x^4, ... or T_0, T_2, T_4, ...), the coefficient lists directly store c_0, c_1, c_2, ... or l_0, l_1, l_2, ...
        The `insert_zeros_loop` static method is used internally to convert these compressed lists into the standard format expected by `numpy.polynomial.Polynomial` and `numpy.polynomial.chebyshev.Chebyshev` objects, which require explicit zeros for missing odd powers.
    * **cheby_coeffs.json:** This file is expected to exist in the same directory as the script and contain pre-computed Chebyshev coefficients. 
        Its structure is assumed to be a dictionary or list where relevant coefficients can be accessed by integer (or string representation of integer) keys/indices.
        (Further detail on its structure should be given under the relevant methods).

    **Usage:**
    To use this class, instantiate it with a positive sigma value, then call methods to compute Taylor or Chebyshev coefficients as needed. 
    """
    def __init__(self, sigma=1):
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        self.sigma = sigma

    def __repr__(self):
        return f"TargetSeriesCoefficients(sigma={self.sigma})"
    
    def __str__(self):
        return f"TargetSeriesCoefficients with sigma={self.sigma}"

    def insert_zeros_loop(self, original_list: list) -> list:
        """
        Inserts a zero after each element in the original list, except for the last element.

        Args:
            original_list (list): The list to insert zeros into.
        Returns:
            list: A new list with zeros inserted after each element of the original list, except for the last element.
        """
        new_list = []
        for i, item in enumerate(original_list):
            new_list.append(item)
            if i < len(original_list) - 1: # Don't add a zero after the last element
                new_list.append(0)
        return new_list

    def TargetFunction(self, x: float) -> float:
        """
        Returns the target function value for a given input x.

        Args:
            x (float): The input value for which to compute the target function.
        Returns:
            float: The computed value of the target function.
        """
        return np.exp( - (1/(2 * self.sigma**2)) * (np.arcsin(x))**2)

    def ArcsineTaylorCoeffs(self, n: int) -> float:
        """
        Takes a non-negative integer n and returns the n-th order coefficient of the Taylor series expansion of arcsine.
            arcsine(x) = \sum_{n=0}^{\infty} s_n x^{2n+1} 
        Args:
            n (int): The index of the coefficient to return.
        Returns:
            float: The n-th coefficient of the Taylor series expansion of arcsine.

        Note: 
            arcsine has an odd order expansion, so coefficient n corresponds to the term of degree 2n+1.
        """
        if n < 0:
            raise ValueError("n must be non-negative.")
        
        return (1/(2**(2*n)))*math.comb(2*n,n)*(1/(2*n+1))
    
    def ArcsineCauchyCoeffs(self, m: int) -> float:
        """
        Computes the m-th order coefficient of the Cauchy product of arcsine with itself.
        The Cauchy product of arcsin(x)^2 is given by:
            arcsin(x) * arcsin(x) = \sum_{m=0}^{\infty} u_m x^{2m+2}
            u_m = \sum_{j=0}^{m} s_j * s_{m-j}.
        Args:
            m (int): The index of the coefficient to return.
        Returns:
            float: The m-th order coefficient of the Cauchy product of arcsine with itself.
        Note:
            attempting to cache this did not yield a significant speedup, so we compute it directly.
        """
        if m < 0:
            raise ValueError("m must be non-negative.")
        total_sum = 0
        for j in range(m+1):  # Loop from j = 0 to m
            first_term = self.ArcsineTaylorCoeffs(j)
            second_term = self.ArcsineTaylorCoeffs(m - j)
            total_sum += first_term * second_term
        
        return total_sum
    
    def ScaledArcsineCauchyCoeffs(self, m: int) -> float:
        """
        Scaled version of the Cauchy coefficients for arcsine, scaled by -1/(2*sigma^2).

        Args:
            m (int): The index of the coefficient to return.
        Returns:
            float: The scaled m-th order coefficient of the Cauchy product of arcsine with itself.
        """
        return (-2*(self.sigma**2))**(-1) * self.ArcsineCauchyCoeffs(m)

    def TargetTaylorCoeff(self, n: int) -> float:
        """
        Computes the coefficient of index n (corresponding to x^{2n}) of the Taylor series expansion of the target function about x=0.
        The coefficients are defined using the n-th complete Bell polynomial:
            c_n = \frac{1}{n!} B_{n}(1! * g(0), 2! * g(1), \dots, (n)! * g(n-1))
            g(m) = -\frac{1}{2\sigma^2} u_m and u_m is the m-th Cauchy coefficient of arcsine.
        The n-th complete Bell polynomial is a sum over n many partial Bell polynomials:
            B_{n}(x_1, x_2, \dots, x_n) = \sum_{k=1}^{n} B_{n,k}(x_1, x_2, \dots, x_n)
        Via the recurrence relation:
            B_n(x_1, \dots, x_n) = \sum_{k=1}^{n} \binom{n-1}{k-1} x_k B_{n-k}(x_1, \dots, x_{n-k}).
        Args:
            n (int): The index of the coefficient to return.
        Returns:
            tuple: A tuple containing the coefficient of index n (corresponding to x^{2n}) and a string representation of the coefficient.
        """
        if n == 0:
            return 1, "0-th coefficient (~ x^0}): c_0 = 1"

        g_vec = [math.factorial(k+1) * self.ScaledArcsineCauchyCoeffs(k) for k in range(n)]
        B = [0.0] * (n + 1)
        B[0] = 1
        for i in range(1, n + 1):
            B[i] = sum(math.comb(i - 1, k - 1) * g_vec[k - 1] * B[i - k] for k in range(1, i + 1))
        result = (1 / math.factorial(n)) * B[n]
        output = f"{n}-th coefficient (~ x^{2*n}): c_{n} = {result}"
        return result, output
    
    def ListTargetTaylorCoeff(self, trunc_index: int, full: bool=False) -> list:
        """
        Produces a list of coefficients for the Taylor series expansion of the target function about x=0, up to the specified trunc_degree.
        Args:
            trunc_index (int): The maximum coefficient index (corresponding to a polynomial of degree 2d) of the Taylor series expansion.
            full (bool): If True, the list will include zeros for odd powers.
        Returns:
            list: A list of coefficients for the Taylor series expansion.
        Note:
            The i-th element of the list corresponds to the coefficient c_i for the term x^(2i).
            All coefficient lists start from c_0 and go up to c_trunc_index.
        ---
        Example:
        (Example list not necessarily representative of actual output, just for illustration)
        >>> D = TargetSeriesCoefficients(sigma=1)
        >>> D.ListTargetTaylorCoeff(3)
        [1,2,3,4]

        -> c_0 = 1, c_1 = 2, c_2 = 3, c_3 = 4
        -> The polynomial is: 1 + 2x^2 + 3x^4 + 4x^6
        """
        output_list = [self.TargetTaylorCoeff(n)[0] for n in range(0, trunc_index + 1)]
        if full==True:
            return self.insert_zeros_loop(output_list)
        else:
            return output_list
    
    def ChebyCoeff(self, degree: int) -> list:
        """
        Reads in the Chebyshev polynomial coefficients from a JSON file and returns the coefficients for the degree-th order Chebyshev polynomial of the first kind.
        Args:
            degree (int): The order of the Chebyshev polynomial to return coefficients for.
        Returns:
            list: A list of coefficients for the Chebyshev polynomial of the first kind of order degree.
        Note:
            The input degree must be a multiple of 2.
            The coefficients are stored in a JSON file named 'cheby_coeffs.json'.
            All coefficient lists start from t_0 and go up to t_degree.
            The largest even degree Chebyshev polynomial is 100, so the maximum degree cannot exceed 100.
        ---
        Example:
        (Example list not necessarily representative of actual output, just for illustration)
        >>> D = TargetSeriesCoefficients(sigma=1)
        >>> D.ChebyCoeff(4)
        [1,-8,8]

        -> t_0 = 1, t_2 = 2, t_4 = 3
        -> The polynomial is: 1 + -8x^2 + 8x^4 = T_4(x)
        """
        if degree < 0 or degree % 2 != 0 or degree > 100:
            raise ValueError("degree must be a non-negative even integer less than or equal to 100.")
        
        with open('cheby_coeffs.json', 'r') as f:
            coeffs = json.load(f)

        return coeffs[int(degree/2)]

    def ListChebyCoeff(self, max_degree: int) -> list:
        """
        Returns a list of coefficient arrays for Chebyshev polynomials of the first kind from degree-0 to degree-max_degree in steps of 2.
        Args:
            max_degree (int): The maximum order of the Chebyshev polynomial.
        Returns:
            list: A list of coefficient arrays for Chebyshev polynomials of the first kind.
        Note:
            The l-th array in the list corresponds to the coefficients for the Chebyshev polynomial of order (2l).
            The i-th element of the l-th array corresponds to the coefficient for x^2i in the Chebyshev polynomial of order (2l).
        ---
        Example:
        (Example list not necessarily representative of actual output, just for illustration)
        >>> D = TargetSeriesCoefficients(sigma=1)
        >>> D.ListChebyCoeff(4)
        [[1], [-1, 2], [1, -8, 8]]

        -> The first array corresponds to T_0(x) = 1
        -> The second array corresponds to T_2(x) = -1 + 2x^2
        -> The third array corresponds to T_4(x) = 1 - 8x^2 + 8x^4
        """
        with open('cheby_coeffs.json', 'r') as f:
            coeffs = json.load(f)

        return coeffs[:max_degree//2 + 1]
    
    def ListTargetChebyCoeff(self, trunc_index: int, full: bool=False) -> list:
        """
        Produces a list of coefficients for the Chebyshev polynomial decomposition of the target function P_d(x).
        \sum_{n=0}^{d} c_n x^{2n} = \sum_{n=0}^{d} l_n T_{2n}(x)
        l_n = \frac{1}{t^{(2n)}_n} (c_n - \sum_{m=n+1}^{d} l_m t^{(2n)}_m)
        where t^{(2n)}_m is the m-th coefficient of the Chebyshev polynomial of order (2n). 
        Args:
            trunc_index (int): The maximum coefficient index (corresponding to the Chebyshev of degree 2d) of the Chebyshev polynomial decomposition.
            full (bool): If True, the list will include zeros for odd powers.
        Returns:
            list: A list of coefficients for the Chebyshev polynomial decomposition of the target function.
        Note:
            The i-th element of the list corresponds to the coefficient for the Chebyshev polynomial of degree (2i)
            All coefficient lists start from l_0 and go up to l_max_index.
        ---
        Example:
        (Example list not necessarily representative of actual output, just for illustration)
        >>> D = TargetSeriesCoefficients(sigma=1)
        >>> D.ListTargetChebyCoeff(4)
        [1,-2,3,-4,5]

        -> l_0 = 1, l_1 = -2, l_2 = 3, l_3 = -4, l_4 = 5
        -> The polynomial is: 1 - 2T_2(x) + 3T_4(x) - 4T_6(x) + 5T_8(x)
        -> This is equivalent to the Taylor series expansion of the target function up to degree 8.
        """
        taylor_coeff_list = self.ListTargetTaylorCoeff(trunc_index)

        if len(taylor_coeff_list) != trunc_index+1:
            raise ValueError(f"taylor_coeff_list must be of length {trunc_index+1}")

        cheby_coeff_list = [0] * (trunc_index+1) # Initialise with zeros

        # Compute the last Chebyshev coefficient
        cheby_coeff_max = taylor_coeff_list[trunc_index] / self.ChebyCoeff(2*trunc_index)[trunc_index]
        cheby_coeff_list[trunc_index] = cheby_coeff_max # Set the last coefficient

        for k in range(trunc_index-1,-1,-1): # Count in reverse
            tail_sum = 0
            for m in range(k+1,trunc_index+1):
                scaled_coeff = cheby_coeff_list[m] * self.ChebyCoeff(2*m)[k]
                tail_sum += scaled_coeff
                
            cheby_coeff_k = (taylor_coeff_list[k] - tail_sum) / self.ChebyCoeff(2*k)[k]
            cheby_coeff_list[k] = cheby_coeff_k # Set the k-th coefficient

        if full==True:
            return self.insert_zeros_loop(cheby_coeff_list)
        else:
            return cheby_coeff_list
    
    def TargetChebyCoeff(self, trunc_index: int, n: int) -> float:
        """
        Computes the coefficient of index n (corresponding to x^{2n}) of the Taylor series expansion of the target function about x=0 in terms of Chebyshev polynomials.
        Args:
            trunc_index (int): The maximum coefficient index of the Chebyshev polynomial decomposition.
            n (int): The index of the coefficient to return.
        Returns:
            float: The coefficient of index n (corresponding to x^{2n}) of the Chebyshev polynomial decomposition of the target function P_d(x).
        Note:
            trunc_index must be a non-negative even integer.
            We must specify trunc_index as this function calls ListTargetChebyCoeff.
        """
        if n < 0:
            raise ValueError(f"{n} must be non-negative.")
        if trunc_index < 0 or trunc_index % 2 != 0:
            raise ValueError(f"{trunc_index} must be a non-negative even integer.")
        
        cheby_coeff_list = self.ListTargetChebyCoeff(trunc_index)
        if len(cheby_coeff_list) <= n:
            raise ValueError(f"n={n} is out of bounds for the computed coefficients.")
        
        return cheby_coeff_list[n]

    def CoeffListToPolynomial(self, coeff_list: list, expansion_type: str) -> Polynomial or chebyshev.Chebyshev:
        """
        Converts a list of coefficients into a Polynomial or Chebyshev polynomial object.
        Args:
            coeff_list (list): List of coefficients.
            expansion_type (str): Type of polynomial, either "TAYLOR" or "CHEBYSHEV".
        Returns:
            Polynomial or Chebyshev object based on the expansion_type.
        Note:
            The coefficients in the list should be in ascending order of powers, i.e., the first element corresponds to the constant term (x^0), the second to the coefficient of x^2, and so on.
            Recall that we are dealing with even powers only, so the coefficients for odd powers will be zero and not included in the initial list.
        """
        # Raise error if there is a zero in the input list
        if any(x == 0 for x in coeff_list):
            raise ValueError("The input list must not contain any zeros.")
        
        if expansion_type == "TAYLOR":
            coeff_list = self.insert_zeros_loop(coeff_list)
            return Polynomial(coeff_list)

        elif expansion_type == "CHEBYSHEV":
            coeff_list = self.insert_zeros_loop(coeff_list)
            return chebyshev.Chebyshev(coeff_list)

        else:
            raise ValueError("expansion_type must be either 'TAYLOR' or 'CHEBYSHEV'")