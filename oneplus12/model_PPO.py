import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from environment import Environment

class PPO(tf.keras.Model):
    def __init__(self, input_dim, output_dim, lr_actor=0.0003, lr_critic=0.001, entropy_coefficient=0.01):
        super(PPO, self).__init__()
        
        # Actor network (policy network)
        self.actor = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(output_dim, activation='softmax')
        ])
        
        # Critic network (value network)
        self.critic = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)  # Value function output
        ])
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.entropy_coefficient = entropy_coefficient

    def call(self, inputs):
        # Forward pass through both actor and critic networks
        return self.actor(inputs), self.critic(inputs)
    
    def select_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        prob = self.actor(state)
        action = np.random.choice(prob.shape[-1], p=prob.numpy()[0])  # Sample based on probability distribution
        return action, prob[0, action].numpy()

    def compute_advantages(self, rewards, values, next_values, gamma=0.99, lam=0.95):
        delta = rewards[0] + gamma * next_values[0] - values[0]
        advantages = []
        advantage = 0.0
        advantage = delta + gamma * lam * advantage
        advantages.append(advantage)
        return list(reversed(advantages))


class PPOAgent:
    def __init__(self, input_dim, output_dim, clip_ratio=0.2, gamma=0.99, lam=0.95, update_epochs=10, batch_size=64, entropy_coefficient=0.01):
        self.model = PPO(input_dim, output_dim, entropy_coefficient=entropy_coefficient)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def train(self, env, episodes=1000):
        state = env.reset()
        for episode in range(episodes):
            episode_rewards, episode_states, episode_actions, episode_probs, episode_values = [], [], [], [], []
            
            # Collect trajectory
            action, prob = self.model.select_action(state)
            next_state, reward = env.step(action, state)
            episode_states.append(state)
            episode_actions.append(action)
            episode_probs.append(prob)
            episode_values.append(self.model.critic(np.expand_dims(state, axis=0))[0][0].numpy())
            episode_rewards.append(reward)
            
            state = next_state

            # Compute next values and advantages
            next_values = episode_values[1:] + [0]  # Assuming terminal state has value 0
            advantages = self.model.compute_advantages(episode_rewards, episode_values, next_values, self.gamma, self.lam)
            returns = [adv + val for adv, val in zip(advantages, episode_values)]

            # Convert to arrays for batching
            episode_states = np.array(episode_states)
            episode_actions = np.array(episode_actions)
            episode_probs = np.array(episode_probs)
            advantages = np.array(advantages, dtype=np.float32)
            returns = np.array(returns, dtype=np.float32)
            
            # Update the actor and critic over multiple epochs
            for _ in range(self.update_epochs):
                indices = np.random.permutation(len(episode_states))
                for i in range(0, len(indices), self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    batch_states = episode_states[batch_indices]
                    batch_actions = episode_actions[batch_indices]
                    batch_probs = episode_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    # Train actor with entropy bonus
                    with tf.GradientTape() as tape:
                        new_probs = self.model.actor(batch_states)
                        action_indices = np.arange(len(batch_actions))
                        selected_probs = tf.gather(new_probs, batch_actions, batch_dims=1)
                        ratios = selected_probs / batch_probs
                        clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                        actor_loss = -tf.reduce_mean(tf.minimum(ratios * batch_advantages, clipped_ratios * batch_advantages))
                        
                        # Entropy regularization to encourage exploration
                        entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
                        actor_loss -= self.model.entropy_coefficient * tf.reduce_mean(entropy)

                    actor_grads = tape.gradient(actor_loss, self.model.actor.trainable_variables)
                    self.model.actor_optimizer.apply_gradients(zip(actor_grads, self.model.actor.trainable_variables))
                    
                    # Train critic
                    with tf.GradientTape() as tape:
                        values = self.model.critic(batch_states)
                        critic_loss = tf.reduce_mean((batch_returns - tf.squeeze(values)) ** 2)

                    critic_grads = tape.gradient(critic_loss, self.model.critic.trainable_variables)
                    self.model.critic_optimizer.apply_gradients(zip(critic_grads, self.model.critic.trainable_variables))

            print(f"Episode {episode}, Total Reward: {sum(episode_rewards)}")

env = Environment()
ppo_agent = PPOAgent(input_dim=10, output_dim=64)
ppo_agent.train(env, episodes=20001)
