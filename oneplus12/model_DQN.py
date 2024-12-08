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

def get_replay_buffer_from_csv(file_path):
    buffer = deque()
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            state = np.array(row[:10], dtype=np.float32)
            action = int(row[20])
            reward = float(row[21])
            next_state = np.array(row[10:20], dtype=np.float32)
            buffer.append((state, action, reward, next_state))
    return buffer

data_file = "data_file.csv"
target_fps = 120
testing =  True
# testing =  False
only_train = False 
only_train = True
view = 'SurfaceView[com.tencent.tmgp.sgame/com.tencent.tmgp.sgame.SGameActivity]\\(BLAST\\)#538'

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.f = open(data_file, "a")
        self.load('yuanshen_60.csv')
        self.load('douyin_30.csv')
        self.load('wangzhe_120.csv')

    def add(self, experience, raw_experience):
        self.buffer.append(experience)
        line = ",".join(map(str, experience[0])) + ',' + ",".join(map(str, experience[3])) + ',' + str(experience[1]) + ',' + str(experience[2]) + ","
        line += ",".join(map(str, raw_experience[0])) + "," + ",".join(map(str, raw_experience[1])) + '\n'
        # print(line)
        self.f.write(line)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

    def size(self):
        return len(self.buffer)

    def load(self, file_path):
        self.buffer += get_replay_buffer_from_csv(file_path)

class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu', input_dim=input_dim)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(output_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.02, buffer_size=500000, batch_size=256):
        self.model = DQN(input_dim, output_dim)
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
        if testing:
            self.model = tf.keras.models.load_model('my_model_yuanshen_douyin_100000')

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
                    continue

                # # 将经验存入ReplayBuffer
                if(int(raw_state[-1]) >= target_fps -2):
                    print('here fps is ok')
                    self.buffer.add((state, action, reward, next_state),(raw_state, raw_next_state))
                    self.buffer.add((state, action, reward, next_state),(raw_state, raw_next_state))

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

                if episode % 5000 == 0:
                    self.model.save(f'my_model_wangzhe_{episode}')
                    print(f"Episode {episode}, Loss: {loss}")

                end = datetime.now()
                print('time:', (end - start).microseconds)
                print(f"Episode {episode}, Loss: {loss} , Epsilon: {epsilon:.3f}")
            else:
                loss = self.train_step()
                if episode % 5000 == 0:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.model.save(f'my_model_yuanshen_douyin_wangzhe_{episode}')
                    print(f"Time: {current_time} , Episode {episode}, Loss: {loss}")
                # start = datetime.now()
                # print('time:', (end - start).microseconds)


env = Environment(view)
env.init_view()
dqn = DQNAgent(10, 625)
if testing or only_train:
    dqn.train(env, episodes=100001, epsilon_start=0, epsilon_end=0.02)
else:
    dqn.train(env, episodes=20001, epsilon_start=0.99, epsilon_end=0.02)

