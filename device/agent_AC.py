import warnings
import os
from threading import Thread
import time
import numpy as np
import random
import socket
import struct
import math
import sys
sys.path.append('/home/wakaba/Desktop/zTT/')
import utils.tools as tools
from datetime import datetime

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
    # from keras import backend as K
    from tensorflow.compat.v1.keras import backend as K
    from collections import defaultdict
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers import Dense
    from collections import deque
    # import tensorflow as tf
    import matplotlib.pyplot as plt

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import socket

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras.optimizers import Adam, RMSprop


def send_socket_data(message, host='127.0.0.1', port=8888):
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


experiment_time = 1000
clock_change_time = 30
cpu_power_limit = 1000
gpu_power_limit = 1600
action_space = 9
target_fps = 60
target_temp = 65
beta = 2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


class ACAgent:
    def __init__(self, state_size, action_size,model_dir='save_model/'):
        self.load_model = True
        self.state_size = state_size
        self.action_size = action_size
        self.actions = list(range(9))
        self.clk_action_list = []
        for i in range(3):
            for j in range(3):
                clk_action = (3 * i + 2, j + 1)
                self.clk_action_list.append(clk_action)

        # Hyperparameters
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.005
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.08
        self.epsilon_min = 0.1
        self.batch_size = 64
        self.train_start = 150

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Actor and Critic models
        self.actor = self.build_actor_model()
        self.critic = self.build_critic_model()

        # Load weights if they exist
        self.load_model = True
        if self.load_model:
            self.load_models()

    def build_actor_model(self):
        input_state = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(input_state)
        x = Dense(64, activation='relu')(x)
        output_actions = Dense(self.action_size, activation='softmax')(x)
        model = Model(inputs=input_state, outputs=output_actions)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.actor_learning_rate))
        return model

    def build_critic_model(self):
        input_state = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(input_state)
        x = Dense(64, activation='relu')(x)
        output_value = Dense(1)(x)
        model = Model(inputs=input_state, outputs=output_value)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.critic_learning_rate))
        return model

    def get_action(self, state):
        state = np.array([state])
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            action_probs = self.actor.predict(state)[0]
            return np.random.choice(self.action_size, p=action_probs)

    def train(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])

        # Get the value of the current state
        value = self.critic.predict(state)[0]

        # Get the target value
        if done:
            target_value = reward
        else:
            next_value = self.critic.predict(next_state)[0]
            target_value = reward + self.discount_factor * next_value

        # Train Critic
        critic_loss = self.critic.train_on_batch(state, np.array([target_value]))

        # Train Actor
        with tf.GradientTape() as tape:
            advantage = target_value - value
            action_probs = self.actor(state, training=True)
            action_prob = action_probs[0, action]
            actor_loss = -tf.math.log(action_prob) * advantage
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        return critic_loss, actor_loss

    def save_models(self):
        # Save actor and critic model weights
        self.actor.save_weights(os.path.join(self.model_dir, 'actor_model.h5'))
        self.critic.save_weights(os.path.join(self.model_dir, 'critic_model.h5'))
        print("Models saved to disk.")

    def load_models(self):
        actor_weights_path = os.path.join(self.model_dir, 'actor_model.h5')
        critic_weights_path = os.path.join(self.model_dir, 'critic_model.h5')

        if os.path.exists(actor_weights_path) and os.path.exists(critic_weights_path):
            self.actor.load_weights(actor_weights_path)
            self.critic.load_weights(critic_weights_path)
            print("Models loaded from disk.")
        else:
            print("No saved models found. Initializing new models.")


def get_reward(fps, power, target_fps, c_t, g_t, c_t_prev, g_t_prev, beta):
    v1 = 0
    v2 = 0

    u = max(1, fps / target_fps)

    if g_t <= target_temp:
        v2 = 0
    else:
        v2 = 2 * (target_temp - g_t)

    if c_t_prev < target_temp:
        if c_t >= target_temp:
            v1 = -2

    if fps >= target_fps:
        u = 1
    else:
        u = math.exp(0.1 * (fps - target_fps))

    return u + v1 + v2 + beta / power


def normalization(big_cpu_freq, little_cpu_freq,big_util, little_util, mem, fps):
    big_cpu_freq = int(big_cpu_freq) / int(cpu_freq_list[1][-1])
    little_cpu_freq = int(little_cpu_freq) / int(cpu_freq_list[0][-1])
    mem = int(mem) / 1e6
    fps = int(fps) / 60
    return big_cpu_freq, little_cpu_freq, float(big_util), float(little_util),int(mem), fps

def get_reward(fps, target_fps, big_clock, little_clock,):
    reward = (fps - target_fps) * 20 +  ( big_clock + little_clock  ) *  (-100) 
    return reward

def process_action(action):
    # print(action)
    action1, action2 = action % 3 , action // 3
    return [0 , action1, action2, 0]
    
if __name__=="__main__":

    # s_dim: 状态维度，即输入的状态向量的维度。
    # h_dim: 隐藏层的维度，即神经网络中间层的神经元数量。
    # branches : 每个分支的action的数量
    agent = ACAgent(state_size=6, action_size=9)

    cpu_freq_list, gpu_freq_list = tools.get_freq_list('k20p')
    little_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[0])
    big_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[1])
    super_big_cpu_clock_list = tools.uniformly_select_elements(6, cpu_freq_list[2])   # 若是没有超大核，则全部为0

    state=(0,0,0,0,0,0)
    action=0
    loss = 0
    experiment_time=1000
    target_fps=25
    reward = 0
    closs,aloss=0,0

    f = open("output.csv", "w")
    f.write(f'episode,big_cpu_freq,little_cpu_freq,big_util,little_util,ipc,cache_miss,fps,action,aloss,closs,reward\n')

    t=1
    try:
        while t < experiment_time:
            t1 = datetime.now()
            temp = send_socket_data('0').split(',')
            big_cpu_freq = temp[0]
            little_cpu_freq =temp[1]
            fps = temp[2]
            mem = temp[3]
            little_util = temp[4]
            big_util = temp[5]
            ipc = temp[6]
            cache_miss = temp[7]
            print(temp)

            f.write(f'{t},{big_cpu_freq},{little_cpu_freq},{big_util},{little_util},{ipc},{cache_miss},{fps},{action},{aloss},{closs},{reward}\n')
            f.flush() 
            # print('[{}] state:{} action:{} fps:{}'.format(t, state,action,fps))
            # print(losses)

            big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps = normalization(big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps)

            # 解析数据
            # next_state=(underlying_data[0], underlying_data[1], underlying_data[2], underlying_data[3], underlying_data[4] ,fps)
            next_state = (big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps)
            
            # reward 
            reward = get_reward(fps, target_fps,big_cpu_freq, little_cpu_freq)
            # print(state, action, next_state, reward)

            # 获得action
            action = agent.get_action(state)
            
            processed_action = process_action(action)
           
            done =1
            res = send_socket_data(f'1,{big_cpu_clock_list[processed_action[1]]},{little_cpu_clock_list[processed_action[2]]}')
            closs, aloss = agent.train(state, action, reward, next_state, done)
            aloss=0
            if(int(res) == -1):
                print('freq set error')
                break
            
            # update some state
            state = next_state
            t += 1

            if t % 60 == 0:
                agent.actor_learning_rate = 0.1
                agent.critic_learning_rate = 0.1
                print('[Reset learning_rate]')

            # 下面这段是记录进文件，供绘图使用的 
            # f.write(f'episode,big_cpu_freq,little_cpu_freq,gpu_freq,cpu_temp,fps\n')


    finally:
        f.close()
 