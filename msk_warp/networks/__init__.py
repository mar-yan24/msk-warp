from msk_warp.networks.actor import ActorStochasticMLP, ActorDeterministicMLP
from msk_warp.networks.critic import CriticMLP

ACTOR_MAP = {
    'ActorStochasticMLP': ActorStochasticMLP,
    'ActorDeterministicMLP': ActorDeterministicMLP,
}

CRITIC_MAP = {
    'CriticMLP': CriticMLP,
}
