"""SHAC (Short Horizon Actor Critic) adapted for MuJoCo Warp.

Adapted from DiffRL's algorithms/shac.py. The core algorithm is unchanged —
the gradient bridge (WarpSimStep) handles the Warp<->PyTorch autodiff interface.
"""

import os
import time
import copy

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tensorboardX import SummaryWriter
import yaml

from msk_warp.envs.cartpole_swing_up import CartPoleSwingUpEnv
from msk_warp.networks.actor import ActorStochasticMLP, ActorDeterministicMLP
from msk_warp.networks.critic import CriticMLP
from msk_warp.utils.common import seeding, print_info
from msk_warp.utils.running_mean_std import RunningMeanStd
from msk_warp.utils.dataset import CriticDataset
from msk_warp.utils.time_report import TimeReport
from msk_warp.utils.average_meter import AverageMeter
import msk_warp.utils.torch_utils as tu


ENV_MAP = {
    'CartPoleSwingUp': CartPoleSwingUpEnv,
}

ACTOR_MAP = {
    'ActorStochasticMLP': ActorStochasticMLP,
    'ActorDeterministicMLP': ActorDeterministicMLP,
}

CRITIC_MAP = {
    'CriticMLP': CriticMLP,
}


class SHAC:
    def __init__(self, cfg):
        env_name = cfg['params']['env']['name']
        env_fn = ENV_MAP[env_name]

        seeding(cfg['params']['general']['seed'])

        self.device = cfg['params']['general']['device']

        self.env = env_fn(
            num_envs=cfg['params']['config']['num_actors'],
            device=self.device,
            episode_length=cfg['params']['env'].get('episode_length', 240),
            stochastic_init=cfg['params']['env'].get('stochastic_init', True),
            substeps=cfg['params']['env'].get('substeps', 4),
            model_path=cfg['params']['env'].get('model_path', 'assets/cartpole.xml'),
            action_strength=cfg['params']['env'].get('action_strength', 20.0),
            no_grad=False,
        )

        print('num_envs =', self.env.num_envs)
        print('num_actions =', self.env.num_actions)
        print('num_obs =', self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length

        self.gamma = cfg['params']['config'].get('gamma', 0.99)

        self.critic_method = cfg['params']['config'].get('critic_method', 'one-step')
        if self.critic_method == 'td-lambda':
            self.lam = cfg['params']['config'].get('lambda', 0.95)

        self.steps_num = cfg['params']['config']['steps_num']
        self.max_epochs = cfg['params']['config']['max_epochs']
        self.actor_lr = float(cfg['params']['config']['actor_learning_rate'])
        self.critic_lr = float(cfg['params']['config']['critic_learning_rate'])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')

        self.target_critic_alpha = cfg['params']['config'].get('target_critic_alpha', 0.4)

        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device)

        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)

        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)

        self.critic_iterations = cfg['params']['config'].get('critic_iterations', 16)
        self.num_batch = cfg['params']['config'].get('num_batch', 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg['params']['config'].get('name', 'CartPole')

        self.truncate_grad = cfg['params']['config']['truncate_grads']
        self.grad_norm = cfg['params']['config']['grad_norm']

        # Per-step observation gradient clipping for BPTT stability.
        # Without a full dynamics Jacobian backward (missing in WarpSimStep),
        # all cross-step gradients flow through the actor network, which can
        # amplify exponentially over the rollout horizon. This clips the
        # gradient at each step boundary, mimicking the damping that a full
        # dynamics backward path would provide.
        self.obs_grad_clip = cfg['params']['config'].get('obs_grad_clip', 0.5)

        self.log_dir = cfg['params']['general']['logdir']
        os.makedirs(self.log_dir, exist_ok=True)

        # Save config
        save_cfg = copy.deepcopy(cfg)
        yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
        self.save_interval = cfg['params']['config'].get('save_interval', 500)

        # Create actor/critic
        actor_name = cfg['params']['network'].get('actor', 'ActorStochasticMLP')
        critic_name = cfg['params']['network'].get('critic', 'CriticMLP')
        self.actor = ACTOR_MAP[actor_name](
            self.num_obs, self.num_actions, cfg['params']['network'], device=self.device
        )
        self.critic = CRITIC_MAP[critic_name](
            self.num_obs, cfg['params']['network'], device=self.device
        )
        self.target_critic = copy.deepcopy(self.critic)

        self.save('init_policy')

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            betas=cfg['params']['config']['betas'],
            lr=self.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            betas=cfg['params']['config']['betas'],
            lr=self.critic_lr,
        )

        # Replay buffers
        self.obs_buf = torch.zeros(
            (self.steps_num, self.num_envs, self.num_obs),
            dtype=torch.float32, device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.done_mask = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.next_values = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.target_values = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.ret = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Counting
        self.iter_count = 0
        self.step_count = 0

        # Episode tracking
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_length_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf

        # Average meters
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        # Timer
        self.time_report = TimeReport()

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros(
            (self.steps_num + 1, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros(
            (self.steps_num + 1, self.num_envs),
            dtype=torch.float32, device=self.device,
        )

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)
            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # Initialize trajectory (cuts gradient graph)
        obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            with torch.no_grad():
                self.obs_rms.update(obs)
            obs = obs_rms.normalize(obs)

        for i in range(self.steps_num):
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, extra_info = self.env.step(torch.tanh(actions))

            with torch.no_grad():
                raw_rew = rew.clone()

            rew = rew * self.rew_scale

            if self.obs_rms is not None:
                with torch.no_grad():
                    self.obs_rms.update(obs)
                obs = obs_rms.normalize(obs)

            # Per-step gradient clipping: prevent exponential amplification
            # through the BPTT chain. The gradient of obs accumulates from all
            # future steps through the actor network; clipping it here bounds
            # the cross-step amplification factor.
            if self.obs_grad_clip is not None and obs.requires_grad:
                _clip = self.obs_grad_clip
                obs.register_hook(
                    lambda g, c=_clip: g * torch.clamp(c / (g.norm() + 1e-8), max=1.0)
                )

            if self.ret_rms is not None:
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            next_values[i + 1] = self.target_critic(obs).squeeze(-1)

            for id in done_env_ids:
                if (
                    torch.isnan(extra_info['obs_before_reset'][id]).sum() > 0
                    or torch.isinf(extra_info['obs_before_reset'][id]).sum() > 0
                    or (torch.abs(extra_info['obs_before_reset'][id]) > 1e6).sum() > 0
                ):
                    next_values[i + 1, id] = 0.0
                elif self.episode_length[id] < self.max_episode_length:
                    next_values[i + 1, id] = 0.0
                else:
                    if self.obs_rms is not None:
                        real_obs = obs_rms.normalize(extra_info['obs_before_reset'][id])
                    else:
                        real_obs = extra_info['obs_before_reset'][id]
                    next_values[i + 1, id] = self.target_critic(real_obs).squeeze(-1)

            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                actor_loss = actor_loss + (
                    -rew_acc[i + 1, done_env_ids]
                    - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]
                ).sum()
            else:
                actor_loss = actor_loss + (
                    -rew_acc[i + 1, :]
                    - self.gamma * gamma * next_values[i + 1, :]
                ).sum()

            gamma = gamma * self.gamma
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()

            # Episode loss tracking
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    for done_env_id in done_env_ids:
                        if self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6:
                            print('ep loss error')
                            raise ValueError
                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.0
                        self.episode_discounted_loss[done_env_id] = 0.0
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.0

        actor_loss /= self.steps_num * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().cpu().item()
        self.step_count += self.steps_num * self.num_envs

        return actor_loss

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                    self.lam * self.gamma * Ai
                    + self.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                )
                Bi = self.gamma * (
                    self.next_values[i] * self.done_mask[i]
                    + Bi * (1.0 - self.done_mask[i])
                ) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi

    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
        target_values = batch_sample['target_values']
        return ((predicted_values - target_values) ** 2).mean()

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    def train(self):
        self.start_time = time.time()

        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")

        self.time_report.start_timer("algorithm")

        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")
            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())

                if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1e6:
                    print('WARNING: NaN or extreme gradient detected (norm={:.2e}), zeroing grads'.format(
                        self.grad_norm_before_clip.item() if not torch.isnan(self.grad_norm_before_clip) else float('nan')))
                    for p in self.actor.parameters():
                        if p.grad is not None:
                            p.grad.zero_()

            self.time_report.end_timer("compute actor loss")
            return actor_loss

        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # Learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # Train actor
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure).detach().item()
            self.time_report.end_timer("actor training")

            # Train critic
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values, drop_last=False)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            for j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()

                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()
                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')

            self.time_report.end_timer("critic training")

            self.iter_count += 1
            time_end_epoch = time.time()

            # Logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar('lr/iter', lr, self.iter_count)
            self.writer.add_scalar('actor_loss/step', self.actor_loss, self.step_count)
            self.writer.add_scalar('actor_loss/iter', self.actor_loss, self.iter_count)
            self.writer.add_scalar('value_loss/step', self.value_loss, self.step_count)
            self.writer.add_scalar('value_loss/iter', self.value_loss, self.iter_count)

            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss

                self.writer.add_scalar('policy_loss/step', mean_policy_loss, self.step_count)
                self.writer.add_scalar('policy_loss/time', mean_policy_loss, time_elapse)
                self.writer.add_scalar('policy_loss/iter', mean_policy_loss, self.iter_count)
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, self.iter_count)
                self.writer.add_scalar('policy_discounted_loss/step', mean_policy_discounted_loss, self.step_count)
                self.writer.add_scalar('policy_discounted_loss/iter', mean_policy_discounted_loss, self.iter_count)
                self.writer.add_scalar('best_policy_loss/step', self.best_policy_loss, self.step_count)
                self.writer.add_scalar('best_policy_loss/iter', self.best_policy_loss, self.iter_count)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)
                self.writer.add_scalar('episode_lengths/time', mean_episode_length, time_elapse)
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            print(
                'iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, '
                'fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.2f}, '
                'grad norm after clip {:.2f}'.format(
                    self.iter_count,
                    mean_policy_loss,
                    mean_policy_discounted_loss,
                    mean_episode_length,
                    self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch),
                    self.value_loss,
                    self.grad_norm_before_clip,
                    self.grad_norm_after_clip,
                )
            )

            self.writer.flush()

            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + "_policy_iter{}_reward{:.3f}".format(self.iter_count, -mean_policy_loss))

            # Update target critic (EMA)
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1.0 - alpha) * param.data)

        self.time_report.end_timer("algorithm")
        self.time_report.report()

        self.save('final_policy')

        # Save histories
        np.save(os.path.join(self.log_dir, 'episode_loss_his.npy'), np.array(self.episode_loss_his))
        np.save(os.path.join(self.log_dir, 'episode_discounted_loss_his.npy'), np.array(self.episode_discounted_loss_his))
        np.save(os.path.join(self.log_dir, 'episode_length_his.npy'), np.array(self.episode_length_his))

        self.close()

    def save(self, filename=None):
        if filename is None:
            filename = 'best_policy'
        torch.save(
            [self.actor, self.critic, self.target_critic, self.obs_rms, self.ret_rms],
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
        self.obs_rms = checkpoint[3].to(self.device) if checkpoint[3] is not None else None
        self.ret_rms = checkpoint[4].to(self.device) if checkpoint[4] is not None else None

    def close(self):
        self.writer.close()
