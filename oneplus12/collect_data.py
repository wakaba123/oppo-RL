import tensorflow as tf
import numpy as np
import random
from collections import deque
from environment import Environment
from tensorflow.keras.models import load_model
import pickle

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


data_file = "data_file.csv"

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.f = open(data_file, "a")

    def add(self, experience):
        self.buffer.append(experience)
        line = ",".join(map(str, experience[0])) + ',' + ",".join(map(str, experience[3])) + ',' + str(experience[1]) + ',' + str(experience[2]) + '\n'
        print(line)
        self.f.write(line)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

    def size(self):
        return len(self.buffer)
    
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

class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.02, buffer_size=500000, batch_size=64):
        self.model = DQN(input_dim, output_dim)
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
        self.target_model = DQN(input_dim, output_dim)

        dummy_input = np.zeros((1, input_dim))
        self.model(dummy_input)
        self.target_model(dummy_input)
        self.target_model.set_weights(self.model.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        # self.loss_fn = tf.keras.losses.Huber()
        
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_step(self):
        if self.buffer.size() < self.batch_size:
            return 0

        # 从ReplayBuffer采样
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_next = self.target_model(next_states)
            q_target = q_values.numpy()
            if(np.any(np.isnan(q_next))):
                print('=-------------=--------------=')
                print(q_next)

            # 计算Q-learning目标
            for i in range(self.batch_size):
                target = rewards[i] + self.gamma * np.max(q_next[i])
                q_target[i][actions[i]] = target
                # print(rewards[i], target)

            if np.any(np.isnan(q_values)) or np.any(np.isnan(q_target)):
                print("Warning: NaN values in q_values or q_target.")
                return 0xffffffff   # 或者返回一个默认的损失值

            loss = self.loss_fn(q_values, q_target)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(625)
        q_values = self.model(np.expand_dims(state, axis=0))
        return np.argmax(q_values.numpy()[0])
    
    # @tf.function(input_signature=[
    #     tf.TensorSpec([64, 10], tf.float32),
    #     tf.TensorSpec([64, 1], tf.float32),
    # ]) 
    # def train_online(self, batch_states, batch_rewards):
    #     with tf.GradientTape() as tape:
    #         prediction = self.model(batch_states)
    #         loss = self.model.loss(batch_rewards, prediction)
    #     gradients = tape.gradient(loss, self.model.trainable_variables)
    #     self.model.optimizer.apply_gradients(
    #         zip(gradients, self.model.trainable_variables))
    #     result = {"loss": loss}
    #     return result

    # @tf.function(input_signature=[
    #     tf.TensorSpec([1, 10], tf.float32),
    # ])
    # def infer(self, x):
    #     pred =self.model(x)
    #     return {
    #         "output": pred
    #     }

    def train(self, env, episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        # SAVED_MODEL_DIR = "dqn_model_episode_10000_action_online"
        # tf.saved_model.save(
        #     self.model,
        #     SAVED_MODEL_DIR,
        #     signatures={
        #         'train_online':
        #             self.train_online.get_concrete_function(),
        #         'infer':
        #             self.infer.get_concrete_function(),
        #     }
        # )
        # converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
        # converter.target_spec.supported_ops = [
        #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        # ]
        # converter.experimental_enable_resource_variables = True
        # tflite_model = converter.convert()
        # open('dqn.tflite', 'wb').write(tflite_model)
        # exit(0)

        epsilon = epsilon_start
        state = env.reset()

        for episode in range(episodes):
            action = self.select_action(state, epsilon)
            next_state, reward = env.step(action, state)  # 移除done

            # # 将经验存入ReplayBuffer
            if(state[-1] * 60>= 55):
                print('here fps is ok')
                self.buffer.add((state, action, reward, next_state))
                # self.buffer.add((state, action, reward, next_state))
                # self.buffer.add((state, action, reward, next_state))
                # self.buffer.add((state, action, reward, next_state))

            self.buffer.add((state, action, reward, next_state))
            # print(reward)

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
dqn = DQNAgent(10, 625)
# dqn.train(env, episodes=20001, epsilon_start=0.99, epsilon_end=0.02)
dqn.train(env, episodes=100001, epsilon_start=0.99, epsilon_end=0.02)

