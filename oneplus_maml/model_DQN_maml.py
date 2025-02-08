import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras import layers, models
import csv
from datetime import datetime  # Import datetime module
import argparse

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

parser = argparse.ArgumentParser(description='MAML Agent')
parser.add_argument('--model_save_path', type=str, default="my_model_weights.h5", help='Path to save the model weights')
parser.add_argument('--load_model', type=bool, default=True, help='Load the model weights symbol')
parser.add_argument('--model_load_path', type=str, default=None, help='Path to load the model weights')
args = parser.parse_args()
model_save_path = args.model_save_path
model_load_path = args.model_load_path
load_model_or_not = args.load_model

task_files = ['douyin_0116_30.csv',"kuaishou_0119_30.csv"]

class ReplayBuffer:
    def __init__(self, max_size, buffer_file):
        self.buffer = deque(maxlen=max_size)
        self.load(buffer_file)

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

class MAMLAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, meta_lr=0.001, gamma=0.02, buffer_size=500000, batch_size=256, model_save_path="meta_model_weights.h5", model_load_path=None):
        self.meta_model = DQN(input_dim, output_dim)
        self.meta_model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
        self.meta_model.build(input_shape=(None, input_dim))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
        self.meta_lr = meta_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.buffers = [ReplayBuffer(buffer_size, file) for file in task_files]  # Replay buffers for tasks
        self.model_save_path = model_save_path
        if model_load_path:
            self.load_model(model_load_path)

    def meta_train(self, meta_episodes=1000, inner_steps=2):
        for episode in range(meta_episodes):
            meta_gradients = []
            total_loss = 0
            for buffer in self.buffers:
                task_model = clone_model(self.meta_model)
                task_model.build((None, 10))  # Ensure the model is built with the correct input shape
                task_model.set_weights(self.meta_model.get_weights())
                for _ in range(inner_steps):
                    loss, gradients = self.train_step(task_model, buffer)
                    if gradients is not None:
                        meta_gradients.append(gradients)
                    total_loss += loss
            if meta_gradients:
                self.apply_meta_gradients(meta_gradients)
            avg_loss = total_loss / (len(self.buffers) * inner_steps)
            if episode % 10 == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{current_time} - Meta Episode {episode},  Avg Loss: {avg_loss}")
            if episode % 100 == 0 and episode != 0:
                self.save_model(self.model_save_path[:-3] + f'_{episode}_' + '.h5')
                print(f"Model weights saved at episode {episode}")

    def train_step(self, model, buffer):
        if buffer.size() < self.batch_size:
            return 0, None
        states, actions, rewards, next_states = buffer.sample(self.batch_size)
        with tf.GradientTape() as tape:
            q_values = model(states)
            q_next = self.meta_model(next_states)
            q_target = q_values.numpy()
            for i in range(self.batch_size):
                target = rewards[i] + self.gamma * np.max(q_next[i])
                q_target[i, actions[i]] = target
            loss = tf.reduce_mean(tf.square(q_values - q_target))
        gradients = tape.gradient(loss, model.trainable_variables)
        return loss, gradients
        
    def apply_meta_gradients(self, meta_gradients):
        mean_gradients = [tf.reduce_mean([grad[i] for grad in meta_gradients], axis=0) for i in range(len(meta_gradients[0]))]
        self.optimizer.apply_gradients(zip(mean_gradients, self.meta_model.trainable_variables))

    def save_model(self, model_save_path):
        self.meta_model.save_weights(model_save_path)

    def load_model(self, model_load_path):
        print('here load model ', model_load_path)
        self.meta_model.build(input_shape=(None, 10))  # Ensure the model is built with the correct input shape
        self.meta_model.load_weights(model_load_path)
        self.meta_model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.meta_lr)
        )

maml_agent = MAMLAgent(10, 625, model_save_path=model_save_path)
# maml_agent = MAMLAgent(10, 625, model_save_path="meta_model_weights_new.h5",model_load_path="meta_model_weights.h5")
maml_agent.meta_train(meta_episodes=1000000)
