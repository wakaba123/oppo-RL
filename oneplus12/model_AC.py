import tensorflow as tf
import numpy as np
from environment import Environment

class ActorCritic:
    def __init__(self, state_dim, action_dim, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Actor network
        self.actor_model = self.build_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        # Critic network
        self.critic_model = self.build_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def select_action(self, state, epsilon=0.1):
        # Choose action using epsilon-greedy for exploration
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_dim)
        state = np.expand_dims(state, axis=0)
        probs = self.actor_model(state)
        # 检查 NaN 并进行归一化处理
        if np.any(np.isnan(probs)):
            print("Warning: Actor model output contains NaN. Replacing NaNs with 0.")
            probs = np.nan_to_num(probs)  # 将 NaN 替换为 0

        # 归一化 probs
        if np.sum(probs) != 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones(self.action_dim) / self.action_dim  # 处理全零的情况
        action = np.random.choice(self.action_dim, p=np.squeeze(probs))
        return action

    def train_step(self, state, action, reward, next_state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        # Expand dims for compatibility with the model
        state = tf.expand_dims(state, axis=0)
        next_state = tf.expand_dims(next_state, axis=0)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Predict value for the current state and next state
            value = tf.squeeze(self.critic_model(state))
            next_value = tf.squeeze(self.critic_model(next_state))
            
            # Compute the target and the advantage
            target = reward + self.gamma * next_value
            advantage = target - value

            # Calculate actor loss (policy gradient with advantage)
            probs = self.actor_model(state)
            epsilon = 1e-10  # A small constant to prevent log(0)
            action_prob = tf.reduce_sum(probs * tf.one_hot(action, self.action_dim))
            actor_loss = -tf.math.log(action_prob + epsilon) * advantage
            
            # Calculate critic loss (MSE)
            critic_loss = tf.square(advantage)
        
        # Backpropagate the gradients
        actor_grads = tape1.gradient(actor_loss, self.actor_model.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic_model.trainable_variables)
        
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        return actor_loss, critic_loss


    def train(self, env, episodes=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        epsilon = epsilon_start
        state = env.reset()

        for episode in range(episodes):
            action = self.select_action(state, epsilon)
            next_state, reward = env.step(action, state)  # 移除done
            
            print(f"Reward: {reward}")

            # 从Buffer中采样训练数据
            actor_loss, critic_loss = self.train_step(state, action, reward, next_state)
            state = next_state

            # 更新 epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # 输出训练信息
            print(f"Episode {episode}, Actor Loss: {actor_loss:.3f}, Critic Loss: {critic_loss:.3f}, Epsilon: {epsilon:.3f}")

            # 保存模型
            if episode % 5000 == 0:
                self.actor_model.save(f"actor_model_episode_{episode}", save_format="tf")
                self.critic_model.save(f"critic_model_episode_{episode}", save_format="tf")

env = Environment()
ac = ActorCritic(10, 64)
ac.train(env)