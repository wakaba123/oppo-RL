#!/usr/bin/env python3

import warnings
import os
from threading import Thread
import time
import random
import socket
import struct
import math
from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import train_utils as train_utils
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('/home/wakaba/Desktop/branch-rl-dvfs')
import utils.tools as tools
import subprocess
from torchsummary import summary
import torch
from torch import nn, optim
from torch.nn import functional as F
from collections import deque
from torch.cuda.amp import GradScaler, autocast
import socket
import pickle
import argparse

def execute(cmd):
    # print(cmd)
    cmds = [ 'su',cmd, 'exit']
    # cmds = [cmd, 'exit']
    obj = subprocess.Popen("adb shell", shell= True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = obj.communicate(("\n".join(cmds) + "\n").encode('utf-8'))
    return info[0].decode('utf-8')

def send_socket_data(message, host='192.168.2.103', port=8888):
    try:
        # 创建一个 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 连接到服务器
        client_socket.connect((host, port))
        # 发送数据
        client_socket.sendall(message.encode('utf-8'))
        # 接收来自服务器的响应
        response = client_socket.recv(1024)
        return response.decode('utf-8')
        
    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        # 关闭连接
        client_socket.close()

class Config:
    def __init__(self):
        self.algo_name = 'DQN'
        self.train_eps = 500
        self.test_eps = 5
        self.max_steps = 100000
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 800
        self.lr = 0.001
        self.gamma = 0.9
        self.seed = random.randint(0, 100)
        self.batch_size = 32
        self.memory_capacity = 100000
        self.hidden_dim = 256
        self.target_update = 4
        self.n_states = None
        self.n_actions = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

class MLP(nn.Module):
    def __init__(self, n_states, n_actions, n_dims=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc3 = nn.Linear(n_dims, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, transitions):
        self.buffer.append(transitions)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size, sequential = False):
        batch_size = min(batch_size, len(self.buffer))
        if sequential: # 获得有序的buffer数据，最近的若干个数据
            rand_index = random.randint(0, len(self.buffer) - batch_size + 1)
            batch = [self.buffer[i] for i in range(rand_index, rand_index + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def size(self):
        return len(self.buffer)

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer dumped to {filename}")

    def load(self, filename):
        """从文件加载数据到缓冲区"""
        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)
        print(f"Replay buffer loaded from {filename}")

class DQN:
    def __init__(self, policy_net, target_net, memory, cfg):
        self.sample_count = 0
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.cfg = cfg
        self.epsilon = cfg.epsilon_start
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.scaler = GradScaler()

    @torch.no_grad()
    def choose_action(self, state, test=False):
        self.sample_count += 1
        self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                       np.exp(-1. * self.sample_count / self.cfg.epsilon_decay)
        if random.uniform(0, 1) > self.epsilon or test:
            state = torch.tensor(np.array(state), device=self.cfg.device, dtype=torch.float32).unsqueeze(0)
            q_value = self.policy_net(state)
            action = q_value.argmax(dim=1).item()
        else:
            action = random.randrange(self.cfg.n_actions)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(np.array(state), device=self.cfg.device, dtype=torch.float32).unsqueeze(0)
        q_value = self.policy_net(state)
        action = q_value.argmax(dim=1).item()
        return action

    def update(self):
        if self.memory.size() < self.cfg.batch_size:
            return 0
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(self.cfg.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.cfg.device, dtype=torch.float32)
        action_batch = torch.tensor(np.array(action_batch), device=self.cfg.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(np.array(reward_batch), device=self.cfg.device, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.cfg.device, dtype=torch.float32)

        with autocast():
            q_value = self.policy_net(state_batch).gather(1, action_batch)
            next_q_value = self.target_net(next_state_batch).max(dim=1)[0].detach()
            expect_q_value = reward_batch + self.cfg.gamma * next_q_value

            # 检查是否存在 NaN
            if torch.isnan(q_value).any():
                print('nan in q_value')
                print(state_batch)
                exit(0)
            
            if torch.isnan(expect_q_value).any():
                print('nan in expect_q_value')
                print(state_batch)
                exit(0)


            loss = F.mse_loss(q_value, expect_q_value.unsqueeze(1))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # 取消梯度缩放后进行梯度裁剪
        self.scaler.unscale_(self.optimizer)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

def get_reward(fps, target_fps, sbig_freq, big_freq, middle_freq, little_freq,):
    reward = -1 * (power_curve_big(big_clock) + power_curve_little(little_clock)) / 200  + min(fps/target_fps, 1)
    return reward

def process_action(action):
    # print(action)
    action1, action2 = action % 8 , action // 8
    return [0 , action1, action2, 0]

def normalize_mem(mem):
    return mem  /  1e5

def normalize_freq(sbig_freq, big_freq, middle_freq, little_freq):
    return sbig_freq / freq_policy7[-1], big_freq / freq_policy5[-1], middle_freq / freq_policy2[-1], little_freq / freq_policy0[-1]

def normalize_util(sbig_util, big_util, middle_util, little_util):
    return sbig_util / 100, big_util / 100, middle_util / 100, little_util / 100

def normalize_fps(fps):
    return fps / target_fps


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

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help='输入输出文件的名称')
    parser.add_argument('-l', '--load_model', type=str, required=False, help='输入要导入的模型的名称')
    parser.add_argument('-s', '--saved_model', type=str, required=False, help='输入要存储的模型的名称')
    args = parser.parse_args()

    output_dir = args.name
    load_model = False
    # load_model = True
    test = False
    # test = True

    # 判断文件夹是否存在
    if not os.path.exists(output_dir):
        # 如果文件夹不存在，则创建它
        os.makedirs(output_dir)
        print(f"文件夹 '{output_dir}' 已创建。")
    else:
        print(f"文件夹 '{output_dir}' 已存在。")


    # -------------------- here after all config -------------------------

    # s_dim: 状态维度，即输入的状态向量的维度。
    # h_dim: 隐藏层的维度，即神经网络中间层的神经元数量。
    # branches : 每个分支的action的数量
    cfg = Config()
    n_states = 6
    n_actions = 256
    cfg.n_actions = n_actions
    cfg.n_states = n_states
   
    load_model_mark = args.load_model
    saved_model_mark = args.saved_model
    output_file = args.name

    if load_model:
        policy_net= torch.jit.load(f'models/policy_net_{load_model_mark}.pt', map_location=cfg.device)
        policy_net.to(cfg.device)
        target_net= torch.jit.load(f'models/target_net_{load_model_mark}.pt', map_location=cfg.device)
        target_net.to(cfg.device)
    else:
        policy_net = torch.jit.script(MLP(n_states, n_actions, cfg.hidden_dim).to(cfg.device)) 
        target_net = torch.jit.script(MLP(n_states, n_actions, cfg.hidden_dim).to(cfg.device)) 

    memory = ReplayBuffer(cfg.memory_capacity)
    agent = DQN(policy_net, target_net, memory, cfg)

    state=(0,0,0,0,0,0)
    action=0
    loss = 0
    experiment_time=5300
    target_fps=30
    reward = 0

    f = open(f"{output_dir}/{output_file}.csv", "w")
    f.write(f'episode,sbig_cpu_freq,big_cpu_freq,middle_cpu_freq,little_cpu_freq,sbig_util,big_util,middle_util,little_util,ipc,cache_miss,fps,action,loss,reward\n')

    t=1
    try:
        while t < experiment_time:
            t1 = datetime.now()
            temp = send_socket_data('0').split(',')
            sbig_cpu_freq = temp[0]
            big_cpu_freq = temp[1]
            middle_cpu_freq =temp[2]
            little_cpu_freq =temp[3]
            fps = temp[4]
            mem = temp[5]
            sbig_util = temp[6]
            big_util = temp[7]
            middle_util = temp[8]
            little_util = temp[9]
            ipc = temp[10]
            cache_miss = temp[11]
            print(temp)

            normal_sbig_cpu_freq, normal_big_cpu_freq, normal_middle_cpu_freq, normal_little_cpu_freq  = normalize_freq(sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq)
            normal_sbig_util, normal_big_util, normal_middle_util, normal_little_util  = normalize_util(sbig_util, big_util, middle_util, little_util)
            normal_mem = normalize_mem(mem)
            normal_fps = normalize_fps(fps)

            # 解析数据
            # next_state=(underlying_data[0], underlying_data[1], underlying_data[2], underlying_data[3], underlying_data[4] ,fps)
            next_state = (normal_big_cpu_freq, normal_little_cpu_freq, normal_big_util, normal_little_util, normal_mem, normal_fps)
            
            # reward 
            reward = get_reward(fps, target_fps, sbig_cpu_freq, big_cpu_freq, middle_cpu_freq, little_cpu_freq)
            # print(state, action, next_state, reward)

            f.write(f'{t},{sbig_cpu_freq},{big_cpu_freq},{middle_cpu_freq},{little_cpu_freq},{sbig_util},{big_util},{middle_util},{little_util},{ipc},{cache_miss},{fps},{action},{loss},{reward}\n')
            f.flush() 

            # replay memory
            if t != 1:
                agent.memory.push((np.array(state), action, reward, np.array(next_state)))

            # 获得action
            action = agent.choose_action(state, test)
            print(action)
            
            target_sbig_freq, target_big_freq, target_middle_freq, target_little_freq = process_action(action)

            tools.set_cpu_freq([little_cpu_clock_list[processed_action[1]], big_cpu_clock_list[processed_action[2]], super_big_cpu_clock_list[processed_action[3]]]) # 若没有超大核，实际超大核不会设置
            res = send_socket_data(f'1,{big_cpu_clock_list[processed_action[1]]},{little_cpu_clock_list[processed_action[2]]}')
            if(int(res) == -1):
                print('freq set error')
                break

            if (t > 5 and t <= experiment_time - 300):
                loss = agent.update()
                print('here loss is ', loss)
            
            # update some state
            state = next_state
            t += 1

            if t % 100 == 0:
                agent.memory.dump('replay_buffer.pkl')

            time.sleep(0.04)


    finally:
        if not test:
            torch.jit.save(agent.policy_net, f'models/policy_net_{saved_model_mark}.pt')
            torch.jit.save(agent.target_net, f'models/target_net_{saved_model_mark}.pt')
        f.close()
 