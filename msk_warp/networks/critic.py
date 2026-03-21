import torch.nn as nn
import numpy as np

from msk_warp.networks import model_utils


class CriticMLP(nn.Module):
    def __init__(self, obs_dim, cfg_network, device='cuda:0'):
        super().__init__()
        self.device = device
        self.layer_dims = [obs_dim] + cfg_network['critic_mlp']['units'] + [1]

        init_ = lambda m: model_utils.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['critic_mlp']['activation']))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)
        self.obs_dim = obs_dim

    def forward(self, observations):
        return self.critic(observations)
