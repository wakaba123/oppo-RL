import random
import numpy as np
import torch
from copy import deepcopy
from torch import nn, optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import train_utils as train_utils
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('/home/wakaba/Desktop/zTT/')
import utils.tools as tools
import Pixel_3a.PMU.pmu as pmu
import Pixel_3a.PowerLogger.powerlogger as powerlogger
from SurfaceFlinger.get_fps import SurfaceFlingerFPS


class Config:
    def __init__(self):
        self.batch_size = 1
        self.memory_capacity = 10000
        self.lr_a = 2e-3
        self.lr_c = 5e-3
        self.gamma = 0.9
        self.tau = 0.005
        self.policy_freq = 2
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.seed = random.randint(0, 100)
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

class ReplayBuffer:
    def __init__(self, cfg):
        self.buffer = np.empty(cfg.memory_capacity, dtype=object)
        self.size = 0
        self.pointer = 0
        self.capacity = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    def push(self, transitions):
        self.buffer[self.pointer] = transitions
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.size = 0
        self.pointer = 0

    def sample(self):
        batch_size = min(self.batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        print(self.buffer[indices])
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.actor_hidden_dim)
        self.fc2 = nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim)
        self.fc3 = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)
        self.action_bound = torch.tensor(cfg.action_bound, dtype=torch.float32, device=cfg.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.action_bound
        return action


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, 1)

        self.fc4 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc5 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc6 = nn.Linear(cfg.critic_hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)

        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(cat))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    @torch.jit.export
    def Q1(self, x, a):
        cat = torch.cat([x, a], dim=1)
        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class TD3:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg)
        self.total_up = 0
        self.scaler = GradScaler()

        self.actor = Actor(cfg).to(cfg.device)
        self.actor = torch.jit.script(self.actor)
        self.actor_target = deepcopy(self.actor)
        self.actor_target = torch.jit.script(self.actor_target)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_a)

        self.critic = Critic(cfg).to(cfg.device)
        self.critic = torch.jit.script(self.critic)
        self.critic_target = deepcopy(self.critic)
        self.critic_target = torch.jit.script(self.critic_target)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_c)
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
    

    @torch.no_grad()
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            self.epsilon *= self.epsilon_decay
            return [random.randrange(0, self.cfg.action_bound)]

        state = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().numpy()
        return action

    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return 0, 0
        self.total_up += 1
        states, actions, rewards, next_states, dones = self.memory.sample()
        actions, rewards, dones = actions.view(-1, 1), rewards.view(-1, 1), dones.view(-1, 1)

        with autocast():
            with torch.no_grad():
                noise = (torch.randn_like(actions, device=self.cfg.device) *
                        self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
                next_actions = (self.actor_target(next_states) + noise).clamp(-self.cfg.action_bound, self.cfg.action_bound)
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = rewards + (1 - dones) * self.cfg.gamma * torch.min(target_q1, target_q2)

            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
        self.critic_optim.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optim)
        self.scaler.update()

        actor_loss = torch.tensor(0.0, device=self.cfg.device)

        if self.total_up % self.cfg.policy_freq == 0:
            for params in self.critic.parameters():
                params.requires_grad = False
            with autocast():
                actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optim)
            self.scaler.update()

            for params in self.critic.parameters():
                params.requires_grad = True
            self.update_params()

        return critic_loss.item(), actor_loss.item()

    def update_params(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)

def get_reward(fps, target_fps, super_big_clock, big_clock, little_clock, gpu_clock, soc_temperature):
    reward = (fps - target_fps) * 20 +  (super_big_clock  + big_clock  + little_clock+ gpu_clock) * -100 - soc_temperature / 20 
    return reward
   

def get_underlying_data():
    gpu_freq = int(tools.get_gpu_freq()) / int(gpu_freq_list[-1])
    cpu_temp = int(tools.get_cpu_temp()) / 100
    gpu_temp = int(tools.get_gpu_temp()) / 100

    temp = tools.get_cpu_freq()
    if len(temp) == 2:
        little_cpu_freq, big_cpu_freq = temp
        little_cpu_freq = int(little_cpu_freq) / int(cpu_freq_list[0][-1])
        big_cpu_freq = int(big_cpu_freq) / int(cpu_freq_list[1][-1])
        return  0, float(big_cpu_freq), float(little_cpu_freq),float(gpu_freq), float(cpu_temp), float(gpu_temp)
    elif len(temp) == 3:
        little_cpu_freq, big_cpu_freq, super_big_cpu_freq = temp
        little_cpu_freq = int(little_cpu_freq) / int(cpu_freq_list[0][-1])
        big_cpu_freq = int(big_cpu_freq) / int(cpu_freq_list[1][-1])
        super_big_cpu_freq = int(super_big_cpu_freq) / int(cpu_freq_list[2][-1])
        return  float(super_big_cpu_freq), float(big_cpu_freq), float(little_cpu_freq),float(gpu_freq), float(cpu_temp), float(gpu_temp)


def process_action(action):
    print(action)
    action = abs(action[0])
    action = round(action)
    action1, action2 = action % 3 , action // 3
    return [0 , action1, action2, 0]


if __name__=="__main__":
    cfg = Config()
    cfg.n_states = 6
    cfg.n_actions = 1
    cfg.action_bound = 9
    agent = TD3(cfg)

    scores, episodes = [], []

    # 下面是一些获取状态所必须的初始化
    # package_name = 'com.bilibili.app.in'  # 该部分需要自己通过top获得当前前台应用的package name并填写
    # pid = tools.get_pid_from_package_name(package_name)
    # current_pmu = pmu.PMUGet(pid)
    # current_pmu.start()
    view = tools.get_view()
    sf_fps_driver = SurfaceFlingerFPS(view)
    sf_fps_driver.start()
    current_power = powerlogger.PowerLogger()
    # last_cpu_util = tools.read_cpu_stats()
    cpu_freq_list, gpu_freq_list = tools.get_freq_list('k20p')
    little_cpu_clock_list = tools.uniformly_select_elements(6, cpu_freq_list[0])
    big_cpu_clock_list = tools.uniformly_select_elements(6, cpu_freq_list[1])
    super_big_cpu_clock_list = tools.uniformly_select_elements(6, cpu_freq_list[2])   # 若是没有超大核，则全部为0

    state=(0,0,0,0,0,0)
    action=[0,0,0,0]
    action_num = 0
    losses = 0
    experiment_time=1000
    target_fps=30

    f = open("output.csv", "w")
    f.write(f'episode,super_big_cpu_freq,big_cpu_freq,little_cpu_freq,gpu_freq,cpu_temp,fps\n')

    t=1
    try:
        while t < experiment_time:
            t1 = datetime.now()
            underlying_data = get_underlying_data() # cpu 所有集群频率， gpu频率， cpu所有集群温度， gpu温度
            power = current_power.getPower() # voltage * current
            # pmu_data = current_pmu.result # pmu 数据[page-faults, task-clock, instructions, cache-references, cache-misses]
            # fps = min(target_fps, sf_fps_driver.getFPS()) # get fps
            fps = sf_fps_driver.getFPS() / target_fps
            # current_cpu_util = tools.read_cpu_stats()
            # cpu_util = tools.calculate_cpu_usage(last_cpu_util, current_cpu_util) # cpu每个核的利用率
            # last_cpu_util = current_cpu_util
            # gpu_util = tools.get_gpu_util() # gpu的利用率
            # print(underlying_data, power, pmu_data, fps, cpu_util,gpu_util)
            t2 = datetime.now()
            print(t2 - t1)

            # 解析数据
            next_state=(underlying_data[0], underlying_data[1], underlying_data[2], underlying_data[3], underlying_data[4] ,fps)
            
            # reward 
            reward = get_reward(fps, target_fps, underlying_data[0], underlying_data[1], underlying_data[2], underlying_data[3], underlying_data[4])

            # replay memory
            # agent.memory.push((torch.tensor(state).float(), torch.tensor(action).float(), torch.tensor(next_state).float(), torch.tensor(reward).float()))
            agent.memory.push((state, action_num, reward, next_state, 0))

            # 获得action
            action_num = agent.choose_action(state)
            action = process_action(action_num)

            # 设置action
            tools.set_gpu_freq(gpu_freq_list[action[0]], action[0])
            tools.set_cpu_freq([little_cpu_clock_list[action[1]], big_cpu_clock_list[action[2]], super_big_cpu_clock_list[action[3]]]) # 若没有超大核，实际超大核不会设置
            c_loss, a_loss = 0, 0

            if (t > 5):
                c_loss, a_loss = agent.update()
            
            # update some state
            state = next_state
            t += 1

             # 下面这段是记录进文件，供绘图使用的 
            f.write(f'{t},{underlying_data[0]},{underlying_data[1]},{underlying_data[2]},{underlying_data[3]},{underlying_data[4]},{fps},{c_loss},{a_loss}\n')
            f.flush() 
            print('[{}] state:{} action:{} next_state:{} reward:{} fps:{}'.format(t, state,action,next_state,reward,fps))


    finally:
        f.close()
 
