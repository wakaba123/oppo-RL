import tensorflow as tf
import numpy as np
import random
from collections import deque
from environment import Environment
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras import layers, models
from datetime import datetime
import pickle
import csv
import time
import os
import argparse  # Import argparse module

def get_replay_buffer_from_csv(file_path):
    buffer = deque()
    with open(file_path, 'r') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            state = np.array(row[:10], dtype=np.float32)
            action = int(row[20])
            reward = float(row[21])
            next_state = np.array(row[10:20], dtype=np.float32)
            buffer.append((state, action, reward, next_state))
    return buffer

parser = argparse.ArgumentParser(description='DQN Agent')
parser.add_argument('--fps', type=int, default=30, help='Frames per second')
parser.add_argument('--model_save_path', type=str, default="my_model_weights.h5", help='Path to save the model weights')
parser.add_argument('--load_model', type=bool, default=False, help='Load the model weights symbol')
parser.add_argument('--model_load_path', type=str, default=None, help='Path to load the model weights')
parser.add_argument('--only_train', type=bool, default=False, help='do not run the environment')
parser.add_argument('--testing', type=bool, default=False, help='test the trained model')
parser.add_argument('--filename', type=str, default="data_file.csv", help='data file name')
args = parser.parse_args()
target_fps = args.fps 
model_save_path = args.model_save_path
model_load_path = args.model_load_path
testing = args.testing
only_train = args.only_train
load_model_or_not = args.load_model
data_file = args.filename
print(data_file)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        if not only_train:
            self.f = open(data_file, "w+")
            if os.path.getsize(data_file) == 0:
                self.f.write('normal_sbig_cpu_freq,normal_big_cpu_freq,normal_middle_cpu_freq,normal_little_cpu_freq,normal_sbig_util,normal_big_util,normal_middle_util,normal_little_util,normal_mem,normal_fps,next_normal_sbig_cpu_freq,next_normal_big_cpu_freq,next_normal_middle_cpu_freq,next_normal_little_cpu_freq,next_normal_sbig_util,next_normal_big_util,next_normal_middle_util,next_normal_little_util,next_normal_mem,next_normal_fps,action,reward,sbig_cpu_freq,big_cpu_freq,middle_cpu_freq,little_cpu_freq,sbig_util,big_util,middle_util,little_util,mem,fps,next_sbig_cpu_freq,next_big_cpu_freq,next_middle_cpu_freq,next_little_cpu_freq,next_sbig_util,next_big_util,next_middle_util,next_little_util,next_mem,next_fps\n')
        # self.load('0119_douyin.csv')
        # self.load('0119_kuaishou.csv')
        # self.load('kuaishou_60.csv')

    def add(self, experience, raw_experience):
        self.buffer.append(experience)
        line = ",".join(map(str, experience[0])) + ',' + ",".join(map(str, experience[3])) + ',' + str(experience[1]) + ',' + str(experience[2]) + ","
        line += ",".join(map(str, raw_experience[0])) + "," + ",".join(map(str, raw_experience[1])) + '\n'
        # print(line)
        if not only_train:
            self.f.write(line)
            self.f.flush()

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

    def size(self):
        return len(self.buffer)

    def load(self, file_path):
        self.buffer += get_replay_buffer_from_csv(file_path)

@tf.keras.saving.register_keras_serializable()
class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu', input_dim=input_dim)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

    def get_config(self):
        config = super(DQN, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.02, buffer_size=500000, batch_size=64, model_save_path="my_model_weights.h5", model_load_path=None):
        self.model = DQN(input_dim, output_dim)
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
        if model_load_path:
            self.load_model(model_load_path)

        # Clone the model for the target model
        self.target_model = DQN(input_dim, output_dim)
        self.target_model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )

        dummy_input = np.zeros((1, input_dim))
        self.model(dummy_input)
        self.target_model(dummy_input)
        self.target_model.set_weights(self.model.get_weights())

        self.output_dim = output_dim
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.model_save_path = model_save_path
        
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_step(self):
        if self.buffer.size() < self.batch_size:
            return 0

        # Sample experience from the ReplayBuffer
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_next = self.target_model(next_states)
            q_target = q_values.numpy()

            # Compute Q-learning target
            for i in range(self.batch_size):
                target = rewards[i] + self.gamma * np.max(q_next[i])
                q_target[i][actions[i]] = target

            loss = self.loss_fn(q_values, q_target)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            # return 0
            return np.random.randint(self.output_dim)
        q_values = self.model(np.expand_dims(state, axis=0))
        return np.argmax(q_values.numpy()[0])
    
    def train(self, env, episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        epsilon = epsilon_start
        state, raw_state = env.reset()

        for episode in range(episodes):
            if not only_train :
                start = datetime.now()
                # time.sleep(0.3)

                action = self.select_action(state, epsilon)
                next_state, reward, raw_next_state = env.step(action, state)  # 移除done
                if testing:
                    state = next_state
                    end = datetime.now()
                    print('time:', (end - start).microseconds)
                    self.buffer.add((state, action, reward, next_state),(raw_state, raw_next_state))
                    continue

                # # 将经验存入ReplayBuffer
                if(int(raw_state[-1]) >= target_fps -2):
                    print('here fps is ok')
                    # self.buffer.add((state, action, reward, next_state),(raw_state, raw_next_state))
                    # self.buffer.add((state, action, reward, next_state),(raw_state, raw_next_state))

                self.buffer.add((state, action, reward, next_state),(raw_state, raw_next_state))

                # 从buffer中训练
                loss = self.train_step()
                if loss > 100  and epsilon < 0.5:
                    epsilon = min(0.5, epsilon + 0.1)
                state = next_state
                raw_state = raw_next_state

                # # 更新 epsilon
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                if episode % 10 == 0:
                    self.update_target_network()

                if episode % 1000 == 0:
                    self.save_model(episode)
                    print(f"Episode {episode}, Loss: {loss}")

                end = datetime.now()
                print('time:', (end - start).microseconds)
                print(f"Episode {episode}, Loss: {loss} , Epsilon: {epsilon:.3f}")
            else:
                loss = self.train_step()
                if episode % 1000 == 0:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.save_model(episode)
                    print(f"Time: {current_time} , Episode {episode}, Loss: {loss}")
                # start = datetime.now()
                # print('time:', (end - start).microseconds)

    def save_model(self, episode):
        self.model.save_weights(self.model_save_path[:-3] + "_" + str(episode) + ".h5")

    def load_model(self, model_load_path):
        self.model.build(input_shape=(None, 10))  # Ensure the model is built with the correct input shape
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        print('model path is ', model_load_path)
        self.model.load_weights(model_load_path)

env = Environment(target_fps, only_train)  # Pass target_fps to Environment
# env.init_view()
dqn = DQNAgent(10, 625, model_save_path=model_save_path, model_load_path=model_load_path)
if testing:
    dqn.train(env, episodes=201, epsilon_start=0, epsilon_end=0.02)
elif only_train:
    dqn.train(env, episodes=2001, epsilon_start=0, epsilon_end=0.02)
else:
    dqn.train(env, episodes=2001, epsilon_start=1, epsilon_end=0.02, epsilon_decay=0.99)

