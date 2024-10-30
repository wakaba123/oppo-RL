import numpy as np
import subprocess
import time
import matplotlib.pyplot as plt

def execute(cmd):
    """Runs an ADB command in the shell and returns the output."""
    obj = subprocess.Popen("adb shell", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return obj.communicate((cmd + "\nexit\n").encode('utf-8'))[0].decode('utf-8')

def reset_stats(policies):
    """Resets the CPU frequency statistics for the given policies."""
    for policy in policies:
        execute(f'echo 1 > /sys/devices/system/cpu/cpufreq/{policy}/stats/reset')

def get_time_in_state(policies):
    """Fetches the time_in_state data for each policy."""
    return [execute(f'cat /sys/devices/system/cpu/cpufreq/{policy}/stats/time_in_state') for policy in policies]

def calculate_power(freqs, powers, time_in_state):
    """Calculates total power consumption based on frequency and time."""
    total_power = 0
    cpu_nums = [2,3,2,1]
    for i, output in enumerate(time_in_state):
        for line in output.splitlines():
            freq, freq_time = map(int, line.split())
            if freq in freqs[i]:
                idx = np.where(freqs[i] == freq)[0][0]
                # print(idx, i)
                total_power += (freq_time / 100) * powers[i][idx] * cpu_nums[i]
    return total_power

# Data for policies
policies = ['policy0', 'policy2', 'policy5', 'policy7']
freqs = [np.array([364800, 460800, 556800, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1344000, 1459200, 1574400, 1689600, 1804800, 1920000, 2035200, 2150400, 2265600]),
         np.array([499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800, 3014400, 3072000, 3148800]),
         np.array([499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800]),
         np.array([480000, 576000, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1363200, 1478400, 1593600, 1708800, 1824000, 1939200, 2035200, 2112000, 2169600, 2246400, 2304000, 2380800, 2438400, 2496000, 2553600, 2630400, 2688000, 2745600, 2803200, 2880000, 2937600, 2995200, 3052800])]
powers = [np.array([4, 5.184, 6.841, 8.683, 10.848, 12.838, 14.705, 17.13, 19.879, 21.997, 25.268, 28.916, 34.757, 40.834, 46.752, 50.616, 56.72, 63.552]),
          np.array([15.386, 19.438, 24.217, 28.646, 34.136, 41.231, 47.841, 54.705, 58.924, 68.706, 77.116, 86.37, 90.85, 107.786, 121.319, 134.071, 154.156, 158.732, 161.35, 170.445, 183.755, 195.154, 206.691, 217.975, 235.895, 245.118, 258.857, 268.685, 289.715, 311.594, 336.845, 363.661]),
          np.array([15.53, 20.011, 24.855, 30.096, 35.859, 43.727, 51.055, 54.91, 64.75, 72.486, 80.577, 88.503, 99.951, 109.706, 114.645, 134.716, 154.972, 160.212, 164.4, 167.938, 178.369, 187.387, 198.433, 209.545, 226.371, 237.658, 261.999, 275.571, 296.108]),
          np.array([31.094, 39.464, 47.237, 59.888, 70.273, 84.301, 97.431, 114.131, 126.161, 142.978, 160.705, 181.76, 201.626, 223.487, 240.979, 253.072, 279.625, 297.204, 343.298, 356.07, 369.488, 393.457, 408.885, 425.683, 456.57, 481.387, 511.25, 553.637, 592.179, 605.915, 655.484])]

# Reset stats and calculate power consumption
output = []
t = 0
while t < 60 :
    reset_stats(policies)
    time.sleep(1)
    time_in_state = get_time_in_state(policies)
    print("=======".join(time_in_state))
    power = calculate_power(freqs, powers, time_in_state)
    # file_power =int(execute('cat /sys/class/power_supply/battery/current_now')) * int (execute('cat /sys/class/power_supply/battery/voltage_now')) / 1000000000
    print(t, power)
    output.append(power)
    t += 1

plt.figure(figsize=(10, 6))
plt.plot(output)
plt.show()

