import torch
import torch.nn as nn

# Check if PyTorch is using a CPU
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
print(f"Using device: {device}")

# # Custom Transformer-based Policy Network
# class ConfigurationTransformer(nn.Module):
#     def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, output_dim=4):
#         super(ConfigurationTransformer, self).__init__()
#         # Input embedding to project the input dimension to the model dimension
#         self.embedding = nn.Linear(input_dim, d_model)
#         # Transformer model
#         self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
#         # Output layer to project the model dimension to the output dimension (4 parameters to configure)
#         self.fc = nn.Linear(d_model, output_dim)

#     def forward(self, src):
#         # Add positional encoding to the input
#         src = self.embedding(src)
#         # Transformer expects input in (sequence_length, batch_size, d_model)
#         # src = src.permute(1, 0, 2)
#         # Pass through transformer
#         transformer_output = self.transformer(src, src)
#         # Get the output for the last step (for autoregressive output)
#         logits = self.fc(transformer_output[-1])
#         return logits

# # Example Usage
# if __name__ == "__main__":
#     # Assuming each input state is a tensor of size (batch_size, input_dim)
#     seq_len = 10
#     batch_size = 4
#     input_dim = 1  # Example input dimension (state representation size)
#     output_dim = 4

#     # Dummy input state for testing
#     input_state = torch.rand(batch_size, input_dim)

#     # Initialize transformer policy network
#     policy_net = ConfigurationTransformer(input_dim=input_dim, output_dim=output_dim)

#     # Forward pass
#     logits = policy_net(input_state)
#     print("Logits:", logits)


# # Environment Configuration (Stub)
# class ConfigurationEnv:
#     def __init__(self, components):
#         self.components = components
#         # Define the observation space, action space, etc.
        
#     def reset(self):
#         # Reset the environment to the initial state
#         return initial_state

#     def step(self, action):
#         # Apply action to environment, return next_state, reward, done, info
#         next_state = None
#         reward = None
#         done = False
#         info = {}
#         return next_state, reward, done, info

# # Transformer-based Policy Network
# class TransformerPolicy(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(TransformerPolicy, self).__init__()
#         config = GPT2Config(vocab_size=100, n_positions=128, n_embd=128, n_layer=2, n_head=2)
#         self.transformer = GPT2Model(config)
#         self.fc = nn.Linear(config.n_embd, output_dim)
    
#     def forward(self, x):
#         x = self.transformer(x)[0]  # Get the last hidden state
#         logits = self.fc(x[:, -1, :])  # Take the last token's output
#         return logits

# # PPO Agent
# class PPOAgent:
#     def __init__(self, env, policy_net, lr=1e-4, gamma=0.99, clip_epsilon=0.2):
#         self.env = env
#         self.policy_net = policy_net
#         self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
#         self.gamma = gamma
#         self.clip_epsilon = clip_epsilon

#     def select_action(self, state):
#         with torch.no_grad():
#             logits = self.policy_net(state)
#             action_probs = torch.softmax(logits, dim=-1)
#             action = torch.multinomial(action_probs, 1)
#         return action.item()

#     def compute_advantages(self, rewards, values, dones):
#         # Compute advantages using GAE or another method
#         advantages = []
#         return advantages

#     def update_policy(self, trajectories):
#         # Extract states, actions, rewards, etc., from trajectories
#         for state, action, reward, next_state, done in trajectories:
#             # Calculate advantages and update the policy
#             pass

#     def train(self, episodes=1000):
#         for episode in range(episodes):
#             state = self.env.reset()
#             done = False
#             trajectories = []
            
#             while not done:
#                 action = self.select_action(state)
#                 next_state, reward, done, _ = self.env.step(action)
#                 trajectories.append((state, action, reward, next_state, done))
#                 state = next_state

#             # Update policy based on collected trajectories
#             self.update_policy(trajectories)

# if __name__ == "__main__":
#     components = [...]  # Define your list of components
#     env = ConfigurationEnv(components)
#     policy_net = TransformerPolicy(input_dim=128, output_dim=4)  # Define input/output dimensions

#     agent = PPOAgent(env, policy_net)
#     agent.train()
