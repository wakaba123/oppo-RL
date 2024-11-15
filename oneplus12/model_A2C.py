import tensorflow as tf
import numpy as np
from environment import Environment

class ActorCritic:
    def __init__(self, state_dim, action_dim, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, entropy_coef=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef  # 增加熵正则化系数

        # 动态学习率调度器
        self.actor_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=actor_lr, decay_steps=1000, decay_rate=0.96
        )
        self.critic_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=critic_lr, decay_steps=1000, decay_rate=0.96
        )
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr_schedule)

        # Actor network
        self.actor_model = self.build_actor()

        # Critic network
        self.critic_model = self.build_critic()

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

    def select_action(self, state, temperature=0.5):
        state = np.expand_dims(state, axis=0)
        probs = self.actor_model(state).numpy()
        probs = np.squeeze(probs)

        # 添加温度参数来增加探索性
        probs = np.power(probs, 1 / temperature)
        probs /= np.sum(probs)
        action = np.random.choice(self.action_dim, p=probs)
        return action

    def train_step(self, states, actions, rewards, next_states):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Critic loss
            values = tf.squeeze(self.critic_model(states))
            next_values = tf.squeeze(self.critic_model(next_states))
            targets = rewards + self.gamma * next_values
            advantages = targets - values
            critic_loss = tf.reduce_mean(tf.square(advantages))

            # Actor loss with entropy regularization
            probs = self.actor_model(states)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_dim), axis=1)
            log_probs = tf.math.log(action_probs + 1e-10)
            actor_loss = -tf.reduce_mean(log_probs * advantages)

            # 计算熵并加入到 Actor 的损失中
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
            actor_loss -= self.entropy_coef * tf.reduce_mean(entropy)

        # Apply gradients
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        return actor_loss, critic_loss

    def train(self, env, episodes=1000, max_steps_per_episode=200):
        for episode in range(episodes):
            state = env.reset()
            states, actions, rewards, next_states = [], [], [], []

            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward = env.step(action, state)

                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)

                state = next_state

                # Train with larger batch size of 64
                if len(states) >= 64:
                    actor_loss, critic_loss = self.train_step(states, actions, rewards, next_states)
                    states, actions, rewards, next_states = [], [], [], []

                    # Logging
                    print(f"Episode {episode}, Step {step}, Actor Loss: {actor_loss:.3f}, Critic Loss: {critic_loss:.3f}")

            # Optional model saving
            if episode % 500 == 0:
                self.actor_model.save(f"actor_model_episode_{episode}", save_format="tf")
                self.critic_model.save(f"critic_model_episode_{episode}", save_format="tf")

env = Environment()
ac = ActorCritic(10, 64)
ac.train(env)
