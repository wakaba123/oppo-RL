import numpy as np
import subprocess
import socket
import time
import bisect
import re

# Remove hardcoded target_fps
# target_fps = 120

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
    # print(sbig_util, big_util, middle_util, little_util)
    return float(sbig_util) / 1, float(big_util) /1, float(middle_util) / 1, float(little_util) / 1

def normalize_fps(fps, target_fps):
    return int(fps) / target_fps
    # return 0

def find_ceil_index(arr, value):
    index = bisect.bisect_left(arr, value)
    res = index if index < len(arr) else -1  
    return res


import numpy as np
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

cpu_nums = [2, 3, 2, 1]
min_power = (power_policy7[0] * cpu_nums[3] +
        power_policy5[0] * cpu_nums[2] +
        power_policy2[0] * cpu_nums[1] +
        power_policy0[0] * cpu_nums[0])  # 116.312

max_power = (power_policy7[-1] * cpu_nums[3] +
        power_policy5[-1] * cpu_nums[2] +
        power_policy2[-1] * cpu_nums[1] +
        power_policy0[-1] * cpu_nums[0]) # 2465.787
 

def get_reward(fps, sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx):
    power = (power_policy7[sbig_freq_idx] * cpu_nums[3] +
             power_policy5[big_freq_idx] * cpu_nums[2] +
             power_policy2[middle_freq_idx] * cpu_nums[1] +
             power_policy0[little_freq_idx] * cpu_nums[0])
    
    reward = 0
    if fps < 0.6:
        fps_reward = 0
    else:
        fps_reward = 2000 * fps
    
    power_reward = power * -1

    reward = fps_reward + power_reward

    # 归一化奖励值
    max_reward = 2000 - min_power  # 根据实际情况设置最大奖励值
    min_reward = 0 - max_power  # 根据实际情况设置最小奖励值
    normalized_reward = (reward - min_reward) / (max_reward - min_reward)
    normalized_reward = 100 * max(0, min(1, normalized_reward))  # 确保归一化后的奖励值在0-1范围内

    return normalized_reward

class Environment:
    def __init__(self, target_fps, training=False):
        # self.target_fps = target_fps // 10 + 1 # Set target_fps as an instance variable
        self.target_fps = target_fps 
        self.name = 'oneplus12'
        self.server_ip = "192.168.2.106"  
        # self.server_ip = "127.0.0.1"  
        self.server_port = 8888
        self.curr_sbig_freq = 0
        self.curr_big_freq = 0
        self.curr_middle_freq = 0
        self.curr_little_freq = 0
        self.training = training
        self.view = self.get_view()
        self.init_view(self.view)
        try:
            self.get_state()
        except:
            print('first time get state error')


    def send_socket_message(self, msg):
        if self.training:
            return ",".join(['0' for i in range(12)])
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.server_ip, self.server_port))
        message = str(msg)  
        client_socket.sendall(message.encode())
        response = client_socket.recv(1024)
        res = response.decode('utf-8')
        client_socket.close()
        return res

    def init_view(self, view):
        self.send_socket_message(f'2,0,0,0,0,{view}')

    def get_state(self):
        temp = self.send_socket_message('0').split(',')
        (sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq,
        fps, mem, sbig_util, big_util, middle_util, little_util,
        ipc, cache_miss) = temp
        self.curr_sbig_freq, self.curr_big_freq, self.curr_middle_freq, self.curr_little_freq = int(sbig_cpu_freq), int(big_cpu_freq), int(middle_cpu_freq), int(little_cpu_freq)
        normal_sbig_cpu_freq, normal_big_cpu_freq, normal_middle_cpu_freq, normal_little_cpu_freq = normalize_freq(sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq)
        normal_sbig_util, normal_big_util, normal_middle_util, normal_little_util = normalize_util(sbig_util, big_util, middle_util, little_util)
        normal_mem = normalize_mem(mem)
        normal_fps = normalize_fps(fps, self.target_fps)
        return (normal_sbig_cpu_freq, normal_big_cpu_freq, normal_middle_cpu_freq, normal_little_cpu_freq,normal_sbig_util, normal_big_util, normal_middle_util, normal_little_util, normal_mem, normal_fps) , (sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq,sbig_util, big_util, middle_util, little_util, mem, fps)
    
    def reset(self):
        if self.training:
            return 0,0
        governor = 'performance'
        freq_policys = [freq_policy0, freq_policy0, freq_policy2, freq_policy2, freq_policy2, freq_policy5, freq_policy5, freq_policy7]
        for i in [0,2,5,7]:
            execute(f'echo {governor} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_governor')
            execute(f'echo {freq_policys[i][-1]} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_max_freq')
            execute(f'echo {freq_policys[i][0]} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_min_freq')
        
        return self.get_state()
        

    def parse_action(self, action): # freqs中存储着利用率算出的频点
        # 每个核有 5 个频点 (0 到 4)，总共有 625 个动作
        num_frequencies = 5
    
        # 计算每个核的频点索引
        sbig_index = (action // (num_frequencies ** 3)) % num_frequencies
        big_index = (action // (num_frequencies ** 2)) % num_frequencies
        middle_index = (action // num_frequencies) % num_frequencies
        little_index = action % num_frequencies

        return int(sbig_index / 5 * len(freq_policy7)) , int(big_index / 5 * len(freq_policy5)), int(middle_index / 5 * len(freq_policy2)), int(little_index / 5 * len(freq_policy0))

    def parse_action2(self, action, state):
        print(action)
        if state[-1] < 0.9:
            target_load = 50
        else: 
            target_load = 70
        sbig_cpu_util , big_cpu_util, middle_cpu_util, little_cpu_util = state[4], state[5], state[6], state[7]
        sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq = self.curr_sbig_freq, self.curr_big_freq, self.curr_middle_freq, self.curr_little_freq
        # print(sbig_cpu_freq, sbig_cpu_util, big_cpu_freq, big_cpu_util, middle_cpu_freq, middle_cpu_util, little_cpu_freq, little_cpu_util)
        
        sbig_target_freq_index = find_ceil_index(freq_policy7,sbig_cpu_freq * 100 * sbig_cpu_util / target_load)
        big_target_freq_index = find_ceil_index(freq_policy5,big_cpu_freq * 100 * big_cpu_util / target_load)
        middle_target_freq_index = find_ceil_index(freq_policy2, middle_cpu_freq * 100 *  middle_cpu_util / target_load)
        little_target_freq_index = find_ceil_index(freq_policy0, little_cpu_freq *100 * little_cpu_util / target_load)
        print(sbig_target_freq_index, big_target_freq_index, middle_target_freq_index, little_target_freq_index)
        # return sbig_target_freq_index, big_target_freq_index, middle_target_freq_index, little_target_freq_index
        # mapping = [-2,-1,0,1,2]
        mapping = [0,0,0,0,0]
        num_frequencies = 5
        # 计算每个核的频点索引
        sbig_index_action = sbig_target_freq_index+ mapping[(action // (num_frequencies ** 3)) % num_frequencies]
        big_index_action = big_target_freq_index+ mapping[(action // (num_frequencies ** 2)) % num_frequencies]
        middle_index_action = middle_target_freq_index+ mapping[(action // num_frequencies) % num_frequencies]
        little_index_action = little_target_freq_index + mapping[action % num_frequencies]

        sbig_index_action = max(0, min(sbig_index_action, len(freq_policy7) - 1))
        big_index_action = max(0, min(big_index_action, len(freq_policy5) - 1))
        middle_index_action = max(0, min(middle_index_action, len(freq_policy2) - 1))
        little_index_action = max(0, min(little_index_action, len(freq_policy0) - 1))

        return (sbig_index_action, big_index_action, middle_index_action, little_index_action)

    def parse_action_for_test(self, action):
        cluster_0 = action % 8
        cluster_2 = action // 8
        return int(cluster_0 * len(freq_policy0) / 8), 0, int(cluster_2 * len(freq_policy5) / 8), 0

        
    def step(self, action, old_state):
        # set action
        # sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx = self.parse_action(action)
        sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx = self.parse_action2(action, old_state)
        # sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx = self.parse_action_for_test(action)
        print(sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx)
        sbig_freq, big_freq, middle_freq, little_freq = freq_policy7[sbig_freq_idx], freq_policy5[big_freq_idx], freq_policy2[middle_freq_idx], freq_policy0[little_freq_idx]
        self.send_socket_message(f"1,{sbig_freq},{big_freq},{middle_freq},{little_freq}")

        time.sleep(0.01)
        # get state
        state, raw_state = self.get_state()
        fps = state[-1] 

        reward = get_reward(fps, sbig_freq_idx, big_freq_idx, middle_freq_idx, little_freq_idx)
        return  state, reward, raw_state
    
    def get_view(self):
        if self.training:
            return ""
        focus_index = [3,6]
        # focus_index= [4,8]
        out = execute('dumpsys SurfaceFlinger | grep -i Explicit -A 30')
        a = out.split('\n')
        view = ""
        for index in focus_index:
            if a[index][-3] == '*':
                view = a[index-1]
                break
        view = view.strip()
        print(f'current view: {view}')

        out = execute('dumpsys SurfaceFlinger --list')
        a = out.split('\n')
        # pattern = r'SurfaceView\[com\.miHoYo\.Yuanshen\/com\..*?\.GetMobileInfo\.MainActivity\]\(BLAST\)#0'
        escaped_text = re.escape(view)
        pattern = escaped_text.replace(re.escape('[...]'), '.*?')

        result = re.findall(pattern, out)

        print(f'current result is {result}')
        return result[0].replace('(','\\(').replace(')','\\)')
