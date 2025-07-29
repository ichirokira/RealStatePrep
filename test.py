# Test

import numpy as np
import CoefficientFinder as cf
import AngleAnalysis as aa
import QSPStatePrep as qsp
import StateCoeffAnalysis as sca
import matplotlib.pyplot as plt

# --- -- -- - - - Test Setup - - -- -- --- #
# Define the sigma and error values for testing
s, err = 1, 1e-7

# Initialise the TargetSeriesCoefficients class with the given sigma
X = cf.TargetSeriesCoefficients(sigma = s)

# Estimate the degree
naive_degree, improved_degree = aa.AnalyticalDegreeFromError(s, err), aa.NumericalDegreeFromError(s, err, 10)
print(f"Naive Degree: {naive_degree}, Improved Degree: {improved_degree}")

# Get the QSP angles
qsp_angles = aa.GetQSPAngleListAdv(s, improved_degree, err, plot = True, verbose = True)
print(f"QSP Angles: {qsp_angles}")

# --- -- -- - - - Compare Numerical Degree - - -- -- --- #
# The bound used in AnalyticalDegreeFromError is loose, it seems that 
#  the degree from NumericalDegreeFromError is between 2 and 1.5 times better!
# We want to extract the ratio to find this rough factor of improvement.
# We need to test for a fixed sigma, then vary the error to see how the degree changes.
# The error can be varied from 1e-7 to 1e-3 over a logarithmic scale and an interval of 100 points.

# Assume 10 qubits.
test_s = 1
error_values = np.logspace(-7, -3, 100)
analytical_degrees = [aa.AnalyticalDegreeFromError(test_s, err) for err in error_values]
numerical_degrees = [aa.NumericalDegreeFromError(test_s, err, 10) for err in error_values]
# Calculate the ratio of numerical to analytical degrees
degree_ratios = np.array(numerical_degrees) / np.array(analytical_degrees)
average_ratio = np.mean(degree_ratios)

# --- -- -- - - - Full Additional Test - - -- -- --- #
aa.GetQSPAngleListAdv(0.5,aa.NumericalDegreeFromError(0.5, 1e-4, 10),1e-4,plot=True, verbose=True)

# --- -- -- - - - Full Coefficient Test - - -- -- --- #
# Here we make use of the test file: shifted_example_angles.txt

sigma, n = 0.5, 4
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
coeff = cf.CoefficientsFromStateVector(vec, [1,0], n)
sca.PlotSignatureVectorElements(coeff, n, sigma)