import configparser
import subprocess
import re
import time
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

def execute(cmd):
    print(cmd)
    cmds = [ 'su',cmd, 'exit']
    # cmds = [cmd, 'exit']
    obj = subprocess.Popen("adb shell", shell= True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = obj.communicate(("\n".join(cmds) + "\n").encode('utf-8'))
    return info[0].decode('utf-8')

# check soc architecture
def check_soc():
    position = '/sys/devices/system/cpu/cpufreq'
    cmd = f'ls {position}'
    result = execute(cmd)
    check_result = result.split()
    return check_result
    

# set cpu governor
def set_cpu_governor(governor):
    cpu_type = check_soc()
    for policy in cpu_type:
        execute(f'echo {governor} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor')

# set cpu frequency
def set_cpu_freq(freq):
    cpu_type = check_soc()
    policy = cpu_type
    for i in range(len(cpu_type)):
        execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_max_freq')
        execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_min_freq')
        execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_min_freq')
        execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_max_freq')
        print(freq[i], execute(f'cat /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_cur_freq').replace('\n',''))
        if not (int(freq[i]) == int(execute(f'cat /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_cur_freq').replace('\n','')) or freq[i] == 0):
            print('here not true')
            execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_max_freq')
            execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_min_freq')
            execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_min_freq')
            execute(f'echo {freq[i]} > /sys/devices/system/cpu/cpufreq/{policy[i]}/scaling_max_freq')


# set cpu frequency by type
def set_cpu_freq_by_type(type, freq): # 0 means little, 1 means big , 2 means super big
    cpu_type = check_soc()
    execute(f'echo {freq} > /sys/devices/system/cpu/cpufreq/{cpu_type[type]}/scaling_min_freq')
    execute(f'echo {freq} > /sys/devices/system/cpu/cpufreq/{cpu_type[type]}/scaling_max_freq')

# get cpu frequency
def get_cpu_freq():
    cpu_type = check_soc()
    # print(cpu_type)
    result = []
    for policy in cpu_type:
        result.append(execute(f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_cur_freq').replace('\n',''))
    return result

# set gpu governor
def set_gpu_governor(governor):
    execute(f'echo {governor} > /sys/class/kgsl/kgsl-3d0/devfreq/governor')

# set gpu frequency
def set_gpu_freq(freq, index):
    execute(f'echo {freq} > /sys/class/kgsl/kgsl-3d0/devfreq/min_freq')
    execute(f'echo {freq} > /sys/class/kgsl/kgsl-3d0/devfreq/max_freq')
    execute(f'echo {index} > /sys/class/kgsl/kgsl-3d0/min_pwrlevel')
    execute(f'echo {index} > /sys/class/kgsl/kgsl-3d0/max_pwrlevel')
    execute(f'echo {freq[:-6]} > /sys/class/kgsl/kgsl-3d0/max_clock_mhz')
    execute(f'echo {freq[:-6]} > /sys/class/kgsl/kgsl-3d0/min_clock_mhz')
    execute(f'echo {index} > /sys/class/kgsl/kgsl-3d0/thermal_pwrlevel')
    execute(f'echo {index} > /sys/class/kgsl/kgsl-3d0/default_pwrlevel')
    
# get gpu frequency
def get_gpu_freq():
    return execute(f'cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq').replace('\n','')

# 目前在ztt中看到的CPU和gpu温度都是通过/sys/devices/virtual/thermal/thermal_zone10/temp获取的，逆天
def get_cpu_temp():
    result = subprocess.run(['adb', 'shell', 'cat', '/sys/class/thermal/thermal_zone10/temp'], capture_output=True, text=True)
    temp = result.stdout
    temp = int(temp)/ 1000
    return temp

def get_gpu_temp():
    result = execute('cat /sys/class/thermal/thermal_zone10/temp')
    return int(result)/1000

def turn_off_on_core(type, on): # type should be 0,1,2   on should be 0 , 1
    cpu_type = check_soc()
    cpu_type[0]='1'
    for i in range(int(cpu_type[type][-1]), 8 if int(type) == 1 else  int(cpu_type[type + 1][-1])):
        execute(f'echo {on} > /sys/devices/system/cpu/cpu{i}/online') 

# get freq list from config file
def get_freq_list(device_name):
    device_name = 'k20p'
    config = configparser.ConfigParser()
    config.read('/home/wakaba/Desktop/zTT/utils/config.ini')

    freq_lists = {
        2: ['little_freq_list','big_freq_list'],
        3: ['little_freq_list', 'big_freq_list', 'super_freq_list']
    }
    cpu_freq_list = [config.get(device_name, freq).split() for freq in freq_lists[2]]
    print(cpu_freq_list)

    cpu_type = check_soc()
    cpu_type = len(cpu_type)
    return 0

    
    if cpu_type in freq_lists:
        cpu_freq_list = [config.get(device_name, freq).split() for freq in freq_lists[cpu_type]]
        if cpu_type == 2:
            cpu_freq_list.append([1,1,1,1,1,1])
    else:
        print("Unsupported CPU type")
        raise ValueError

    gpu_freq_list = config.get(device_name,'gpu_freq_list').split()
    return cpu_freq_list, gpu_freq_list

def uniformly_select_elements(n, array):
    """
    从给定的数组中均匀选择 n 个元素。
    
    参数:
    n (int): 要选择的元素个数。
    array (list): 输入的数组。

    返回:
    list: 包含 n 个均匀分布的元素的新数组。
    """
    if n <= 0 or not array:
        return []

    step = len(array) / n
    selected_elements = [array[int(i * step)] for i in range(n)]
    return selected_elements

def get_view():
    focus_index = [3,6]
    # focus_index= [4,8]
    out = execute('dumpsys SurfaceFlinger | grep -i focus -A 10')
    a = out.split('\n')
    view = ""
    for index in focus_index:
        if a[index][-2] == '*':
            # view = a[index-3]
            view = a[index-1]
            break
    view = view.strip()
    print(f'current view:{view}')

    out = execute('dumpsys SurfaceFlinger --list')
    a = out.split('\n')
    # pattern = r'SurfaceView\[com\.miHoYo\.Yuanshen\/com\..*?\.GetMobileInfo\.MainActivity\]\(BLAST\)#0'
    escaped_text = re.escape(view)
    pattern = escaped_text.replace(re.escape('[...]'), '.*?')

    result = re.findall(pattern, out)

    print(f'current result is {result}')
    return re.escape(result[0])


# 获得各核的利用率，下面两个函数是含状态的

def read_cpu_stats():
    a = execute('cat /proc/stat')
    lines = a.split('\n')
    
    cpu_stats = {}
    for line in lines:
        if line.startswith('cpu'):
            parts = line.split()
            if len(parts) < 5:
                continue
            cpu_id = parts[0]
            user, nice, system, idle, iowait, irq, softirq, steal = map(int, parts[1:9])
            idle_time = idle + iowait
            total_time = user + nice + system + idle + iowait + irq + softirq + steal
            cpu_stats[cpu_id] = (idle_time, total_time)
    return cpu_stats

def calculate_cpu_usage(prev_stats, curr_stats):
    usage = {}
    for cpu in curr_stats.keys():
        prev_idle, prev_total = prev_stats[cpu]
        curr_idle, curr_total = curr_stats[cpu]
        
        idle_delta = curr_idle - prev_idle
        total_delta = curr_total - prev_total
        
        if total_delta == 0:
            usage[cpu] = 0.0
        else:
            usage[cpu] = 100.0 * (1.0 - (idle_delta / total_delta))
    return usage

def get_pid_from_package_name(pacakge_name):
    temp = execute(f'ps -ef').splitlines()
    for line in temp:
        pid = line.split()[1]
        if line.split()[-1] == pacakge_name:
            print(line)
            return pid

def get_gpu_util():
    a = execute('cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage')
    return a

def recover():
    set_cpu_governor('schedutil')
    big_freq_list = [710400,2419200]
    little_freq_list = [300000,1785600]
    execute(f'echo {big_freq_list[-1]} > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq')
    execute(f'echo {big_freq_list[0]} > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq')
    execute(f'echo {little_freq_list[-1]} > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq')
    execute(f'echo {little_freq_list[0]} > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq')
    execute(f'echo 0 > /sys/devices/system/cpu/cpu7/online')
    subprocess.run('adb forward tcp:8888 tcp:8888', shell=True)
    execute('stop vendor.thermal-engine')

def test_oneplus():
    governor = 'uag'
    freq_policys = [freq_policy0, freq_policy0, freq_policy2, freq_policy2, freq_policy2, freq_policy5, freq_policy5, freq_policy7]
    for i in [0,2,5,7]:
        execute(f'echo {governor} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_governor')
        execute(f'echo {freq_policys[i][-1]} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_max_freq')
        execute(f'echo {freq_policys[i][0]} > /sys/devices/system/cpu/cpufreq/policy{i}/scaling_min_freq')
            
if __name__ == '__main__':
    test_oneplus()
    # recover()
    # get_cpu_temp()
    # get_gpu_temp()
    # get_cpu_freq()
    # get_gpu_freq()
    # get_view()
    # get_cpu_times()
    # temp = get_pid_from_package_name('com.bilibili.app.in')
    # print(temp)

    # get_cpu_freq()
    # set_cpu_governor('userspace')
    # set_cpu_governor('performance')
    # big_freq_list=[710400,940800,1171200,1401600]
    # little_freq_list=[576000,768000,844800,1036800,1113600,1305600,1632000,1785600]
    # for big_freq in big_freq_list:
    #     for little_freq in little_freq_list:
    #         set_cpu_freq([little_freq, big_freq, 0])
    #         print(get_cpu_freq())
    #         time.sleep(2)
    # set_cpu_freq([576000, 710400,0 ])
    # print(get_cpu_freq())
    #         time.sleep(2)
    # set_gpu_governor('performance')