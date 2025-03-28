import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.amp import autocast
# from torch.cuda.amp import GradScaler
from torch import autocast
from torch.cuda.amp import GradScaler
import math


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.1):
        super(CustomDecoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention layer
        # tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt2 = F.scaled_dot_product_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention layer
        # tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt2 = F.scaled_dot_product_attention(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward layer
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=64, dropout=0.1):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Actor(nn.Module):
    def __init__(self, num_components, num_panels, device, params):
        super(Actor, self).__init__()
        self.num_components = num_components
        self.state_dim = self.num_components * 4
        self.position_actions = 51
        self.rotation_actions = 24
        self.panel_actions = 2 * num_panels
        self.nhead = 2
        self.dense_dim = 64  # Hidden dimension for the transformer
        self.device = device
        self.scaler = GradScaler()
        self.clip_ratio = params[2]

        # Input encoder (linear embedding)
        self.encoder = nn.Linear(2, self.dense_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=self.dense_dim)

        # Transformer decoder layers
        # decoder_layer = nn.TransformerDecoderLayer(d_model=self.dense_dim, nhead=self.nhead, batch_first=True)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.transformer_decoder = CustomTransformerDecoder(d_model=self.dense_dim, nhead=self.nhead, num_layers=4)

        # Output layers for each action type
        self.position_output_layer = nn.Linear(self.dense_dim, self.position_actions)
        self.rotation_output_layer = nn.Linear(self.dense_dim, self.rotation_actions)
        self.panel_output_layer = nn.Linear(self.dense_dim, self.panel_actions)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params[6])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def generate_square_subsequent_mask(self, sz):
        """Generates a mask to prevent the model from looking at future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, inputs, act=0):
        with autocast(device_type=self.device.type, dtype=torch.float16):
        # with autocast(dtype=torch.float16):
            actions = torch.fmod(torch.arange(0, inputs.size()[1]), 4) + 1
            actions = actions.repeat(inputs.size()[0], 1).to(inputs.device)  # Ensure it runs on the same device (e.g., 'cuda')

            # Concatenate inputs and actions, then pass through the encoder
            x = torch.cat((inputs.unsqueeze(-1), actions.unsqueeze(-1)), dim=-1)
            x = self.encoder(x)

            # Add positional encoding
            x = self.positional_encoding(x)

            # Create a causal mask (prevents future information from leaking)
            seq_len = x.size(1)
            batch_size = x.size(0)
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
            # mask = mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)

            # Pass through the transformer decoder (decoder-only architecture)
            memory = x  # For a decoder-only architecture, we can feed `memory` as the encoded input itself.
            x = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask)

            # Process output based on `act` flag
            if act == 0:  # Panel prediction
                x = self.panel_output_layer(x)
            elif act == 1:  # X location prediction
                x = self.position_output_layer(x)
            elif act == 2:  # Y location prediction
                x = self.position_output_layer(x)
            elif act == 3:  # Rotation prediction
                x = self.rotation_output_layer(x)

            x = F.softmax(x, dim=-1)

        return x

    def sample_configuration(self, observations, act):
        if len(observations[0]) == 0:
            observations = [[0] for x in range(len(observations))]
        input_observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
        output = self(input_observations, act=act)

        output_last = output[:, -1, :]
        log_probs = torch.log(output_last + 1e-10)

        samples = torch.distributions.categorical.Categorical(logits=log_probs).sample()
        action_ids = samples.squeeze()
        action_probs = log_probs.gather(1, action_ids.unsqueeze(1)).squeeze()

        return action_probs, action_ids, output_last

    def ppo_update(self, observation_tensor, action_tensor, logprob_tensor, advantage_tensor):

        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
        # with autocast(dtype=torch.float16):

            # clip_ratio = 0.1
            
            # Separate observation tensor for panel, position, and rotation
            panel_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 0]
            x_position_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 1]
            y_position_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 2]
            rotation_observation_tensor = observation_tensor[torch.arange(len(observation_tensor)) % 4 == 3]

            # Separate action tensor for panel, position, and rotation
            panel_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 0].long()
            x_position_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 1].long()
            y_position_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 2].long()
            rotation_action_tensor = action_tensor[torch.arange(len(action_tensor)) % 4 == 3].long()

            # Compute predicted log probabilities for panel, position, and rotation actions
            panel_pred_probs = self(panel_observation_tensor, act=0)[:,-1,:]
            panel_pred_log_probs = torch.log(panel_pred_probs + 1e-10)
            panel_log_probs = torch.sum(
                F.one_hot(panel_action_tensor, num_classes=self.panel_actions) * panel_pred_log_probs, dim=-1
            )

            # Compute predicted log probabilities for x position actions
            x_position_pred_probs = self(x_position_observation_tensor, act=1)[:,-1,:]
            x_position_pred_log_probs = torch.log(x_position_pred_probs + 1e-10)
            x_position_log_probs = torch.sum(
                F.one_hot(x_position_action_tensor, num_classes=self.position_actions) * x_position_pred_log_probs, dim=-1
            )

            # Compute predicted log probabilities for y position actions
            y_position_pred_probs = self(y_position_observation_tensor, act=2)[:,-1,:]
            y_position_pred_log_probs = torch.log(y_position_pred_probs + 1e-10)
            y_position_log_probs = torch.sum(
                F.one_hot(y_position_action_tensor, num_classes=self.position_actions) * y_position_pred_log_probs, dim=-1
            )

            rotation_pred_probs = self(rotation_observation_tensor, act=3)[:,-1,:]
            rotation_pred_log_probs = torch.log(rotation_pred_probs + 1e-10)
            rotation_log_probs = torch.sum(
                F.one_hot(rotation_action_tensor, num_classes=self.rotation_actions) * rotation_pred_log_probs, dim=-1
            )

            # Reshape and concatenate log probabilities
            log_probs = torch.cat([panel_log_probs.unsqueeze(-1), x_position_log_probs.unsqueeze(-1), 
                                y_position_log_probs.unsqueeze(-1), rotation_log_probs.unsqueeze(-1)], dim=-1)
            log_probs = log_probs.view(-1)

            # Calculate the loss
            ratio = torch.exp(log_probs - logprob_tensor)
            min_advantage = torch.where(
                advantage_tensor > 0,
                (1 + self.clip_ratio) * advantage_tensor,
                (1 - self.clip_ratio) * advantage_tensor
            )
            policy_loss = -torch.mean(torch.min(torch.t(ratio * torch.t(advantage_tensor)), min_advantage))

        # Calculate gradients and update the policy
        # self.optimizer.zero_grad()
        # policy_loss.backward()
        # self.optimizer.step()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()

        # Calculate the KL divergence
        kl = torch.mean(logprob_tensor - log_probs)

        return policy_loss.item(), kl.item()

class Critic(nn.Module):
    def __init__(self, num_components, device, params):
        super(Critic, self).__init__()
        self.num_components = num_components
        self.state_dim = self.num_components * 4
        self.dense_dim = 64  # Hidden dimension for the transformer
        self.num_objectives = 5  # Number of value predictions (e.g., one for each objective)
        self.nhead = 2
        self.device = device
        self.scaler = GradScaler()

        # Input encoder (linear embedding)
        self.encoder = nn.Linear(2, self.dense_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=self.dense_dim)

        # Transformer decoder layers
        # decoder_layer = nn.TransformerDecoderLayer(d_model=self.dense_dim, nhead=8, batch_first=True)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.transformer_decoder = CustomTransformerDecoder(d_model=self.dense_dim, nhead=self.nhead, num_layers=4)

        # Output layer for value estimation
        self.value_output_layer = nn.Linear(self.dense_dim, self.num_objectives)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params[6])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def generate_square_subsequent_mask(self, sz):
        """Generates a mask to prevent the model from looking at future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, inputs):
        with autocast(device_type=self.device.type, dtype=torch.float16):
        # with autocast(dtype=torch.float16):

            actions = torch.fmod(torch.arange(0, inputs.size()[1]), 4) + 1
            actions = actions.repeat(inputs.size()[0], 1).to(inputs.device)  # Ensure it runs on the same device (e.g., 'cuda')

            # Concatenate inputs and actions, then pass through the encoder
            x = torch.cat((inputs.unsqueeze(-1), actions.unsqueeze(-1)), dim=-1)
            x = self.encoder(x)

            # Add positional encoding
            x = self.positional_encoding(x)

            # Create a causal mask (prevents future information from leaking)
            seq_len = x.size(1)
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

            # Pass through the transformer decoder (decoder-only architecture)
            memory = x  # For a decoder-only architecture, we can feed `memory` as the encoded input itself.
            x = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask)

            # Pass through the value output layer (single output for value prediction)
            x = self.value_output_layer(x)

            # Return only the last value prediction, which is typically the value of the current state
            x = x[:, -1, :]

        return x

    def sample_critic(self, observations):
        input_observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
        output = self(input_observations)
        
        return output

    def ppo_update(self, observation, return_buffer, weights):
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
        # with autocast(dtype=torch.float16):
            pred_values = self(observation)
            pred_reward = torch.sum(-pred_values * weights, dim=-1)
            value_loss = torch.mean((return_buffer - pred_reward) ** 2)

        # Backpropagation
        # self.optimizer.zero_grad()
        # value_loss.backward()
        # self.optimizer.step()
        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()

        return value_loss.item()