import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from msk_warp.networks import model_utils


class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super().__init__()
        self.device = device
        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))

        self.actor = nn.Sequential(*modules).to(device)
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def get_logstd(self):
        return None

    def forward(self, observations, deterministic=False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super().__init__()
        self.device = device
        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))

        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = cfg_network.get('actor_logstd_init', -1.0)
        self.logstd = nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32, device=device) * logstd
        )

        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        if deterministic:
            return mu
        std = self.logstd.exp()
        dist = Normal(mu, std)
        return dist.rsample()

    def forward_with_dist(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()
        if deterministic:
            return mu, mu, std
        dist = Normal(mu, std)
        return dist.rsample(), mu, std
