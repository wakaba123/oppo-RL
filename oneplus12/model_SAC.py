import tensorflow as tf
import numpy as np
from environment import Environment

class SAC:
    def __init__(self, state_dim, action_dim, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, alpha_lr=1e-4, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor (policy) network
        self.actor_model = self.build_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        # Critic (Q-function) networks
        self.critic_model_1 = self.build_critic()
        self.critic_model_2 = self.build_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Target critic networks
        self.target_critic_model_1 = self.build_critic()
        self.target_critic_model_2 = self.build_critic()

        # Soft Q Target smoothing (tau)
        self.update_target_networks()

        # Alpha (temperature) parameter
        self.alpha = tf.Variable(0.2, dtype=tf.float32, trainable=False)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='tanh')  # tanh to bound action within [-1, 1]
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_dim + self.action_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def update_target_networks(self):
        # Soft update of target Q networks
        for target_params, model_params in zip(self.target_critic_model_1.trainable_variables, self.critic_model_1.trainable_variables):
            target_params.assign(self.tau * model_params + (1 - self.tau) * target_params)
        for target_params, model_params in zip(self.target_critic_model_2.trainable_variables, self.critic_model_2.trainable_variables):
            target_params.assign(self.tau * model_params + (1 - self.tau) * target_params)

    def select_action(self, state, epsilon=0.1):
        state = np.expand_dims(state, axis=0)
        # Output probabilities for each action using softmax
        logits = self.actor_model(state)
        probs = tf.nn.softmax(logits)  # Softmax to get action probabilities
        action = np.random.choice(self.action_dim, p=probs.numpy().squeeze())  # Sample an action based on probabilities
        
        # Epsilon-greedy exploration: with probability epsilon, choose a random action
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        
        return action

    def train_step(self, state, action, reward, next_state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)  # 这里将 action 改为整数类型
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
    
        # Expand dims to ensure compatibility for concatenation
        state = tf.expand_dims(state, axis=0)  # Adding batch dimension
        next_state = tf.expand_dims(next_state, axis=0)  # Adding batch dimension
        action = tf.expand_dims(action, axis=-1)  # Expand action to make it a column vector
        
        # Convert action to one-hot encoding (size [batch_size, action_dim])
        action_one_hot = tf.one_hot(action, self.action_dim)
    
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Concatenate state and action (one-hot encoded) for both current and next state
            state_action = tf.concat([state, action_one_hot], axis=-1)
            next_state_action = tf.concat([next_state, action_one_hot], axis=-1)
    
            # Predict value for the current state and next state
            value = tf.squeeze(self.critic_model_1(state_action))
            next_value = tf.squeeze(self.target_critic_model_1(next_state_action))
    
            # Compute the target and the advantage
            target = reward + self.gamma * next_value
            advantage = target - value
    
            # Calculate actor loss (policy gradient with advantage)
            logits = self.actor_model(state)  # Get the logits (unnormalized action probabilities)
            probs = tf.nn.softmax(logits)  # Convert logits to probabilities
            action_prob = tf.reduce_sum(probs * action_one_hot)  # Calculate probability of taken action
            actor_loss = -tf.math.log(action_prob + 1e-10) * advantage  # Clip for numerical stability
    
            # Calculate critic loss (MSE)
            critic_loss = tf.square(advantage)
    
            # Calculate entropy loss for exploration
            entropy_loss = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
    
        # Backpropagate the gradients
        actor_grads = tape1.gradient(actor_loss, self.actor_model.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic_model_1.trainable_variables)
    
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model_1.trainable_variables))
    
        # Return all four values
        return actor_loss, critic_loss, critic_loss, entropy_loss

    def train(self, env, episodes=1000, max_timesteps=10000):
        state = env.reset()
        for episode in range(episodes):
            total_reward = 0
            for timestep in range(max_timesteps):
                action = self.select_action(state)
                print(timestep, action)
                next_state, reward = env.step(action, state)  # Removing done
                actor_loss, critic_loss_1, critic_loss_2, entropy_loss = self.train_step(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Actor Loss: {actor_loss:.3f}, Critic Loss 1: {critic_loss_1:.3f}, Critic Loss 2: {critic_loss_2:.3f}, Entropy Loss: {entropy_loss:.3f}")

            if episode % 5000 == 0:
                self.actor_model.save(f"actor_model_episode_{episode}", save_format="tf")
                self.critic_model_1.save(f"critic_model_1_episode_{episode}", save_format="tf")
                self.critic_model_2.save(f"critic_model_2_episode_{episode}", save_format="tf")

env = Environment()
sac = SAC(state_dim=10, action_dim=64)
sac.train(env)
