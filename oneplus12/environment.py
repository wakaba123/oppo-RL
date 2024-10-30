import numpy as np
import subprocess
import socket
import time

target_fps = 60

def execute(cmd):
    print(cmd)
    cmds = [ 'su',cmd, 'exit']
    # cmds = [cmd, 'exit']
    obj = subprocess.Popen("adb shell", shell= True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = obj.communicate(("\n".join(cmds) + "\n").encode('utf-8'))
    return info[0].decode('utf-8')

def normalize_mem(mem):
    return int(mem) / 1e5

def normalize_freq(sbig_freq, big_freq, middle_freq, little_freq):
    return int(sbig_freq) / freq_policy7[-1], int(big_freq) / freq_policy5[-1], int(middle_freq) / freq_policy2[-1], int(little_freq) / freq_policy0[-1]

def normalize_util(sbig_util, big_util, middle_util, little_util):
    return float(sbig_util) / 100, float(big_util) / 100, float(middle_util) / 100, float(little_util) / 100

def normalize_fps(fps):
    return int(fps) / target_fps


freq_policy0 = np.array([364800, 460800, 556800, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1344000, 1459200, 1574400, 1689600, 1804800, 1920000, 2035200, 2150400, 2265600])
power_policy0 = np.array([4, 5.184, 6.841, 8.683, 10.848, 12.838, 14.705, 17.13, 19.879, 21.997, 25.268, 28.916, 34.757, 40.834, 46.752, 50.616, 56.72, 63.552])

# Data for policy2
freq_policy2 = np.array([499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800, 3014400, 3072000, 3148800])
power_policy2 = np.array([15.386, 19.438, 24.217, 28.646, 34.136, 41.231, 47.841, 54.705, 58.924, 68.706, 77.116, 86.37, 90.85, 107.786, 121.319, 134.071, 154.156, 158.732, 161.35, 170.445, 183.755, 195.154, 206.691, 217.975, 235.895, 245.118, 258.857, 268.685, 289.715, 311.594, 336.845, 363.661])

# Data for policy5
freq_policy5 = np.array([499200, 614400, 729600, 844800, 960000, 1075200, 1190400, 1286400, 1401600, 1497600, 1612800, 1708800, 1824000, 1920000, 2035200, 2131200, 2188800, 2246400, 2323200, 2380800, 2438400, 2515200, 2572800, 2630400, 2707200, 2764800, 2841600, 2899200, 2956800])
power_policy5 = np.array([15.53, 20.011, 24.855, 30.096, 35.859, 43.727, 51.055, 54.91, 64.75, 72.486, 80.577, 88.503, 99.951, 109.706, 114.645, 134.716, 154.972, 160.212, 164.4, 167.938, 178.369, 187.387, 198.433, 209.545, 226.371, 237.658, 261.999, 275.571, 296.108])

# Data for policy7
freq_policy7 = np.array([480000, 576000, 672000, 787200, 902400, 1017600, 1132800, 1248000, 1363200, 1478400, 1593600, 1708800, 1824000, 1939200, 2035200, 2112000, 2169600, 2246400, 2304000, 2380800, 2438400, 2496000, 2553600, 2630400, 2688000, 2745600, 2803200, 2880000, 2937600, 2995200, 3052800])
power_policy7 = np.array([31.094, 39.464, 47.237, 59.888, 70.273, 84.301, 97.431, 114.131, 126.161, 142.978, 160.705, 181.76, 201.626, 223.487, 240.979, 253.072, 279.625, 297.204, 343.298, 356.07, 369.488, 393.457, 408.885, 425.683, 456.57, 481.387, 511.25, 553.637, 592.179, 605.915, 655.484])

def get_reward(fps, sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx):
    reward =  ( 1509.68 - power_policy7[sbig_freq_idx] + power_policy5[big_freq_idx] + power_policy2[middle_freq_idx] + power_policy0[little_freq_idx] ) / 10  + 30 * (fps - target_fps) 
    return reward


class Environment:
    def __init__(self):
        self.name = 'oneplus12'
        self.server_ip = "192.168.2.103"  
        self.server_port = 8888

    def send_socket_message(self, msg):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.server_ip, self.server_port))
        message = str(msg)  
        client_socket.sendall(message.encode())
        response = client_socket.recv(1024)
        res = response.decode('utf-8')
        client_socket.close()
        return res

    def get_state(self):
        temp = self.send_socket_message('0').split(',')
        (sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq,
        fps, mem, sbig_util, big_util, middle_util, little_util,
        ipc, cache_miss) = temp
        normal_sbig_cpu_freq, normal_big_cpu_freq, normal_middle_cpu_freq, normal_little_cpu_freq = normalize_freq(sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq)
        normal_sbig_util, normal_big_util, normal_middle_util, normal_little_util = normalize_util(sbig_util, big_util, middle_util, little_util)
        normal_mem = normalize_mem(mem)
        normal_fps = normalize_fps(fps)
        return (normal_sbig_cpu_freq, normal_big_cpu_freq, normal_middle_cpu_freq, normal_little_cpu_freq,normal_sbig_util, normal_big_util, normal_middle_util, normal_little_util, normal_mem, normal_fps)
    
    def reset(self):
        governor = 'performance'
        freq_policys = [freq_policy0, freq_policy0, freq_policy2, freq_policy2, freq_policy2, freq_policy5, freq_policy5, freq_policy7]
        for i in [0,2,5,7]:
            execute(f'echo {governor} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_governor')
            execute(f'echo {freq_policys[i][-1]} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_max_freq')
            execute(f'echo {freq_policys[i][0]} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_min_freq')
        
        return self.get_state()
        

    def parse_action(self, action):
        # 每个核有 5 个频点 (0 到 4)，总共有 625 个动作
        num_frequencies = 5
    
        # 计算每个核的频点索引
        sbig_index = (action // (num_frequencies ** 3)) % num_frequencies
        big_index = (action // (num_frequencies ** 2)) % num_frequencies
        middle_index = (action // num_frequencies) % num_frequencies
        little_index = action % num_frequencies
    
        return int(sbig_index / 5 * len(freq_policy7)) , int(big_index / 5 * len(freq_policy5)), int(middle_index / 5 * len(freq_policy2)), int(little_index / 5 * len(freq_policy0))


    def step(self, action):
        # set action
        sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx = self.parse_action(action)
        print(sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx)
        sbig_freq, big_freq, middle_freq, little_freq = freq_policy7[sbig_freq_idx], freq_policy5[big_freq_idx], freq_policy2[middle_freq_idx], freq_policy0[little_freq_idx]
        self.send_socket_message(f"1,{sbig_freq},{big_freq},{middle_freq},{little_freq}")

        time.sleep(0.01)
        # get state
        state = self.get_state()
        fps = state[-1] * target_fps

        reward = get_reward(fps, sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx)
        return  state, reward

        
            