import tensorflow as tf
import numpy as np
import random
from collections import deque
from environment import Environment
import csv

def get_replay_buffer_from_csv(file_path):
    buffer = deque()
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            state = np.array(row[:10], dtype=np.float32)
            action = int(row[-2])
            reward = float(row[-1])
            next_state = np.array(row[10:20], dtype=np.float32)
            buffer.append((state, action, reward, next_state))
    return buffer

data_file = "data_file.csv"

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.f = open(data_file, "a")
        # self.load('temp2.csv')

    def add(self, experience):
        self.buffer.append(experience)
        line = ",".join(map(str, experience[0])) + ',' + ",".join(map(str, experience[3])) + ',' + str(experience[1]) + ',' + str(experience[2]) + '\n'
        self.f.write(line)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

    def size(self):
        return len(self.buffer)

    def load(self, file_path):
        self.buffer = get_replay_buffer_from_csv(file_path)

class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_dim,))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

class DDQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.99, buffer_size=500000, batch_size=64):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.output_dim = output_dim
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
        
        # 初始化网络
        dummy_input = np.zeros((1, input_dim))
        self.model(dummy_input)
        self.target_model(dummy_input)
        self.update_target_network()
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_step(self):
        if self.buffer.size() < self.batch_size:
            return 0

        # 从ReplayBuffer采样
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_target = q_values.numpy()
            
            # 使用当前Q网络选择下一步的动作
            q_next_online = self.model(next_states)
            q_next_target = self.target_model(next_states)

            for i in range(self.batch_size):
                # DDQN的目标Q值：使用在线网络选择动作，目标网络计算Q值
                best_next_action = np.argmax(q_next_online[i])
                target = rewards[i] + self.gamma * q_next_target[i][best_next_action]
                q_target[i][actions[i]] = target

            loss = self.loss_fn(q_values, q_target)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.output_dim)  # Use self.output_dim instead of self.model.output_shape
        q_values = self.model(np.expand_dims(state, axis=0))
        return np.argmax(q_values.numpy()[0])
    
    def train(self, env, episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        epsilon = epsilon_start
        state = env.reset()

        for episode in range(episodes):
            action = self.select_action(state, epsilon)
            next_state, reward = env.step(action, state) 

            # 将经验存入ReplayBuffer
            self.buffer.add((state, action, reward, next_state))

            # 从buffer中训练
            loss = self.train_step()
            state = next_state

            # 更新 epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            if episode % 10 == 0:
                self.update_target_network()

            if episode % 5000 == 0:
                self.model.save(f"dqn_model_episode_{episode}", save_format="tf")

            print(f"Episode {episode}, Loss: {loss} , Epsilon: {epsilon:.3f}")

env = Environment()
ddqn = DDQNAgent(10, 64)
ddqn.train(env, episodes=20001, epsilon_start=0.99, epsilon_end=0.02)
