import tensorflow as tf
import numpy as np
import random
from collections import deque
from environment import Environment

class ActorNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.state_input = tf.keras.layers.InputLayer(input_shape=(state_dim,))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.action_output = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, inputs):
        x = self.state_input(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.action_output(x)


class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.state_input = tf.keras.layers.InputLayer(input_shape=(state_dim,))
        self.action_input = tf.keras.layers.InputLayer(input_shape=(action_dim,))
        
        # 拼接 state 和 action
        self.concat = tf.keras.layers.Concatenate()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.q_value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        state_input, action_input = inputs
        x = self.concat([state_input, action_input])
        x = self.dense1(x)
        x = self.dense2(x)
        return self.q_value(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.001, buffer_size=100000, batch_size=64, learning_rate=0.001):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.actor_target.set_weights(self.actor.get_weights())

        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_target.set_weights(self.critic.get_weights())

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def update_target_network(self, target_model, model):
        target_weights = target_model.get_weights()
        model_weights = model.get_weights()

        new_weights = [self.tau * model_weights[i] + (1 - self.tau) * target_weights[i] for i in range(len(target_weights))]
        target_model.set_weights(new_weights)


    def select_action(self, state, epsilon=0.1):
        # 确保 state 是一个 numpy 数组
        state = np.array(state)

        if np.random.rand() < epsilon:
            # 返回随机动作
            return np.random.uniform(0, 64, size=(self.action_dim,))  # 返回随机动作，适应状态的维度
        # 使用 actor 网络来生成动作
        action = self.actor(np.expand_dims(state, axis=0))
        print(action)
        return np.squeeze(action, axis=0)


    def update_network(self, state, action, reward, next_state):
        # 扩展 action 的维度，使其与 state 的维度匹配
        action = np.expand_dims(action, axis=1)  # action 变为 (1, 1)
        
        # 获取当前的 Q 值（critic 网络）
        with tf.GradientTape() as tape:
            current_q_value = self.critic([np.expand_dims(state, axis=0), action])

        # 获取下一个状态的 Q 值（target network）
        next_action = self.actor_target(np.expand_dims(next_state, axis=0))
        next_q_value = self.critic_target([np.expand_dims(next_state, axis=0), next_action])

        # 计算目标 Q 值
        target_q_value = reward  * self.gamma * next_q_value

        # 计算损失并更新
        with tf.GradientTape() as tape:
            current_q_value = self.critic([np.expand_dims(state, axis=0), action])
            critic_loss = tf.reduce_mean(tf.square(target_q_value - current_q_value))

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # 使用 critic 的目标来更新 actor 网络
        with tf.GradientTape() as tape:
            actions = self.actor(np.expand_dims(state, axis=0))
            actor_loss = -tf.reduce_mean(self.critic([np.expand_dims(state, axis=0), actions]))

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # 更新目标网络
        self.update_target_network(self.actor_target, self.actor)
        self.update_target_network(self.critic_target, self.critic)

    def train(self, env, episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        epsilon = epsilon_start
        state = env.reset()

        for episode in range(episodes):
            total_reward = 0
            # 选择动作
            action = self.select_action(state, epsilon)
            next_state, reward= env.step(action, state)

            # 存储经验
            self.buffer.append((state, action, reward, next_state))

            # 训练网络
            if len(self.buffer) >= self.batch_size:
                batch = random.sample(self.buffer, self.batch_size)
                for experience in batch:
                    self.update_network(*experience)

            total_reward += reward
            state = next_state

            # 更新 epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

            # 定期保存模型
            # if episode % 100 == 0:
            #     self.actor.save(f"actor_model_episode_{episode}", save_format="tf")
            #     self.critic.save(f"critic_model_episode_{episode}",save_format="tf")


# 假设 env 已经实现并符合上述接口
env = Environment()  # 你需要确保 `env` 有 `reset()` 和 `step()` 方法

state_dim = 10  # 假设 state 有 10 个维度
action_dim = 1  # 假设动作是连续值，且为标量
ddpg_agent = DDPGAgent(state_dim, action_dim)

# 训练 DDPG agent
ddpg_agent.train(env, episodes=20001)
