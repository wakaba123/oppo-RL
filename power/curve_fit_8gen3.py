import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the exponential fitting function
def exp_fit(x, a, b, c):
    return a * np.power(x, b) + c

# Data for policy0
freq_policy0 = np.array([364800, 460800, 556800, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1344000, 1459200, 1574400, 1689600, 1804800, 1920000, 2035200, 2150400, 2265600])
freq_policy0 = freq_policy0 / 1000
power_policy0 = np.array([4, 5.184, 6.841, 8.683, 10.848, 12.838, 14.705, 17.13, 19.879, 21.997, 25.268, 28.916, 34.757, 40.834, 46.752, 50.616, 56.72, 63.552])

# Perform the curve fitting for policy0
params_policy0, _ = curve_fit(exp_fit, freq_policy0, power_policy0)

# Data for policy2
freq_policy2 = np.array([499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800, 3014400, 3072000, 3148800])
freq_policy2 = freq_policy2/  1000
power_policy2 = np.array([15.386, 19.438, 24.217, 28.646, 34.136, 41.231, 47.841, 54.705, 58.924, 68.706, 77.116, 86.37, 90.85, 107.786, 121.319, 134.071, 154.156, 158.732, 161.35, 170.445, 183.755, 195.154, 206.691, 217.975, 235.895, 245.118, 258.857, 268.685, 289.715, 311.594, 336.845, 363.661])

# Perform the curve fitting for policy2
params_policy2, _ = curve_fit(exp_fit, freq_policy2, power_policy2)

# Data for policy5
freq_policy5 = np.array([499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800])
power_policy5 = np.array([15.53, 20.011, 24.855, 30.096, 35.859, 43.727, 51.055, 54.91, 64.75, 72.486, 80.577, 88.503, 99.951, 109.706, 114.645, 134.716, 154.972, 160.212, 164.4, 167.938, 178.369, 187.387, 198.433, 209.545, 226.371, 237.658, 261.999, 275.571, 296.108])
freq_policy5 = freq_policy5/  1000

# Perform the curve fitting for policy5
params_policy5, _ = curve_fit(exp_fit, freq_policy5, power_policy5)

# Data for policy7
freq_policy7 = np.array([480000, 576000, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1363200, 1478400, 1593600, 1708800, 1824000, 1939200, 2035200, 2112000, 2169600, 2246400, 2304000, 2380800, 2438400, 2496000, 2553600, 2630400, 2688000, 2745600, 2803200, 2880000, 2937600, 2995200, 3052800])
power_policy7 = np.array([31.094, 39.464, 47.237, 59.888, 70.273, 84.301, 97.431, 114.131, 126.161, 142.978, 160.705, 181.76, 201.626, 223.487, 240.979, 253.072, 279.625, 297.204, 343.298, 356.07, 369.488, 393.457, 408.885, 425.683, 456.57, 481.387, 511.25, 553.637, 592.179, 605.915, 655.484])
freq_policy7 = freq_policy7 / 1000

# Perform the curve fitting for policy7
params_policy7, _ = curve_fit(exp_fit, freq_policy7, power_policy7)

# Plotting the results
x_vals = np.linspace(364800/ 1000, 3148800 / 1000, 500)

plt.figure(figsize=(10, 6))

# Plot for policy0
plt.plot(freq_policy0, power_policy0, 'bo', label="Policy0 Data")
plt.plot(x_vals, exp_fit(x_vals, *params_policy0), 'b-', label="Policy0 Fit")

# Plot for policy2
plt.plot(freq_policy2, power_policy2, 'go', label="Policy2 Data")
plt.plot(x_vals, exp_fit(x_vals, *params_policy2), 'g-', label="Policy2 Fit")

# Plot for policy5
plt.plot(freq_policy5, power_policy5, 'ro', label="Policy5 Data")
plt.plot(x_vals, exp_fit(x_vals, *params_policy5), 'r-', label="Policy5 Fit")

# Plot for policy7
plt.plot(freq_policy7, power_policy7, 'mo', label="Policy7 Data")
plt.plot(x_vals, exp_fit(x_vals, *params_policy7), 'm-', label="Policy7 Fit")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (W)')
plt.title('Exponential Curve Fitting for Different Policies')
plt.legend()
plt.grid(True)
plt.show()

print(params_policy0, params_policy2, params_policy5, params_policy7)
