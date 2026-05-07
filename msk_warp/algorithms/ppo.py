"""PPO (Proximal Policy Optimization) for MuJoCo Warp environments.

Used as a bootstrap phase to escape standing local minima before handing
off to SHAC for differentiable fine-tuning.  Runs the environment in
no_grad mode (no differentiable physics).

Checkpoint format is SHAC-compatible:
    [actor, critic, target_critic, obs_rms, ret_rms]
so the trained policy can be loaded directly by SHAC.load().
"""

import os
import copy
import time

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tensorboardX import SummaryWriter
import yaml

from msk_warp.envs import ENV_MAP
from msk_warp.networks import ACTOR_MAP, CRITIC_MAP
from msk_warp.utils.common import seeding, print_info
from msk_warp.utils.running_mean_std import RunningMeanStd
from msk_warp.utils.time_report import TimeReport
from msk_warp.utils.average_meter import AverageMeter


class PPO:
    def __init__(self, cfg):
        env_name = cfg['params']['env']['name']
        env_fn = ENV_MAP[env_name]

        seeding(cfg['params']['general']['seed'])

        self.device = cfg['params']['general']['device']

        # Build env kwargs — force no_grad=True for PPO
        env_cfg = cfg['params']['env']
        shac_only_keys = {'name'}
        env_kwargs = {k: v for k, v in env_cfg.items() if k not in shac_only_keys}
        if 'num_actors' in env_kwargs:
            env_kwargs['num_envs'] = env_kwargs.pop('num_actors')
        env_kwargs['device'] = self.device
        env_kwargs['no_grad'] = True

        self.env = env_fn(**env_kwargs)

        print('num_envs =', self.env.num_envs)
        print('num_actions =', self.env.num_actions)
        print('num_obs =', self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length

        self.gamma = cfg['params']['config'].get('gamma', 0.99)
        self.gae_lambda = cfg['params']['config'].get('gae_lambda', 0.95)
        self.steps_num = cfg['params']['config']['steps_num']
        self.max_epochs = cfg['params']['config']['max_epochs']
        self.actor_lr = float(cfg['params']['config']['actor_learning_rate'])
        self.critic_lr = float(cfg['params']['config']['critic_learning_rate'])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'constant')

        self.clip_range = float(cfg['params']['config'].get('clip_range', 0.2))
        self.entropy_coef = float(cfg['params']['config'].get('entropy_coef', 0.01))
        self.value_coef = float(cfg['params']['config'].get('value_coef', 0.5))
        self.ppo_epochs = int(cfg['params']['config'].get('ppo_epochs', 10))
        self.num_minibatches = int(cfg['params']['config'].get('num_minibatches', 32))
        self.normalize_advantages = cfg['params']['config'].get('normalize_advantages', True)
        self.max_grad_norm = float(cfg['params']['config'].get('max_grad_norm', 0.5))
        self.truncate_grad = cfg['params']['config'].get('truncate_grads', True)

        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device)

        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)

        self.name = cfg['params']['config'].get('name', 'PPO')
        self.log_dir = cfg['params']['general']['logdir']
        os.makedirs(self.log_dir, exist_ok=True)

        save_cfg = copy.deepcopy(cfg)
        yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
        self.save_interval = cfg['params']['config'].get('save_interval', 500)

        # Create actor/critic (same architecture as SHAC for checkpoint compat)
        actor_name = cfg['params']['network'].get('actor', 'ActorStochasticMLP')
        critic_name = cfg['params']['network'].get('critic', 'CriticMLP')
        self.actor = ACTOR_MAP[actor_name](
            self.num_obs, self.num_actions, cfg['params']['network'], device=self.device
        )
        self.critic = CRITIC_MAP[critic_name](
            self.num_obs, cfg['params']['network'], device=self.device
        )

        self.betas = tuple(cfg['params']['config'].get('betas', [0.9, 0.999]))
        self._build_optimizers()

        # Rollout buffer
        self.buf_obs = torch.zeros(
            (self.steps_num, self.num_envs, self.num_obs),
            dtype=torch.float32, device=self.device,
        )
        self.buf_actions = torch.zeros(
            (self.steps_num, self.num_envs, self.num_actions),
            dtype=torch.float32, device=self.device,
        )
        self.buf_log_probs = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.buf_rewards = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.buf_dones = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.buf_values = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.buf_advantages = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.buf_returns = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )

        # Episode tracking
        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_loss_his = []
        self.episode_length_his = []
        self.best_policy_loss = np.inf
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        # Current observation carried across rollouts (set during train init)
        self._current_obs = None

        self.iter_count = 0
        self.step_count = 0
        self.time_report = TimeReport()

    def _build_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), betas=self.betas, lr=self.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), betas=self.betas, lr=self.critic_lr,
        )

    @torch.no_grad()
    def collect_rollout(self):
        """Collect steps_num transitions and compute GAE advantages."""
        obs = self._current_obs

        if self.obs_rms is not None:
            obs_rms_snapshot = copy.deepcopy(self.obs_rms)

        for step in range(self.steps_num):
            if self.obs_rms is not None:
                self.obs_rms.update(obs)
                obs_norm = obs_rms_snapshot.normalize(obs)
            else:
                obs_norm = obs

            # Sample action and compute log_prob in one pass
            pre_tanh, mu, std = self.actor.forward_with_dist(obs_norm)
            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
            value = self.critic(obs_norm).squeeze(-1)

            action = torch.tanh(pre_tanh)

            # Step environment
            obs_new, rew, done, extras, _, _ = self.env.step(action)

            # Store transition
            self.buf_obs[step] = obs_norm
            self.buf_actions[step] = pre_tanh
            self.buf_log_probs[step] = log_prob
            self.buf_rewards[step] = rew
            self.buf_dones[step] = done.float()
            self.buf_values[step] = value

            # Episode tracking
            self.episode_length += 1
            self.episode_loss -= rew
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                self.episode_length_meter.update(self.episode_length[done_env_ids])
                for eid in done_env_ids:
                    self.episode_loss_his.append(self.episode_loss[eid].item())
                    self.episode_length_his.append(self.episode_length[eid].item())
                    self.episode_loss[eid] = 0.0
                    self.episode_length[eid] = 0

            obs = obs_new

        # Bootstrap value for last observation
        if self.obs_rms is not None:
            obs_norm = obs_rms_snapshot.normalize(obs)
        else:
            obs_norm = obs
        last_value = self.critic(obs_norm).squeeze(-1)

        # Compute GAE
        self._compute_gae(last_value)

        # Carry current obs forward to next rollout
        self._current_obs = obs

        self.step_count += self.steps_num * self.num_envs

    @torch.no_grad()
    def _compute_gae(self, last_value):
        """Generalized Advantage Estimation."""
        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        next_value = last_value

        for step in reversed(range(self.steps_num)):
            not_done = 1.0 - self.buf_dones[step]
            delta = (
                self.buf_rewards[step]
                + self.gamma * next_value * not_done
                - self.buf_values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            self.buf_advantages[step] = gae
            self.buf_returns[step] = gae + self.buf_values[step]
            next_value = self.buf_values[step]

    def update(self):
        """PPO clipped surrogate update over collected rollout."""
        total_samples = self.steps_num * self.num_envs
        batch_size = total_samples // self.num_minibatches

        # Flatten buffers: (steps, envs, ...) -> (steps*envs, ...)
        flat_obs = self.buf_obs.reshape(total_samples, -1)
        flat_actions = self.buf_actions.reshape(total_samples, -1)
        flat_log_probs = self.buf_log_probs.reshape(total_samples)
        flat_advantages = self.buf_advantages.reshape(total_samples)
        flat_returns = self.buf_returns.reshape(total_samples)

        if self.normalize_advantages:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                idx = indices[start:end]

                mb_obs = flat_obs[idx]
                mb_actions = flat_actions[idx]
                mb_old_log_probs = flat_log_probs[idx]
                mb_advantages = flat_advantages[idx]
                mb_returns = flat_returns[idx]

                # Actor loss
                new_log_probs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = self.critic(mb_obs).squeeze(-1)
                value_loss = (new_values - mb_returns).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss for logging, but update separately
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                (actor_loss + self.entropy_coef * entropy_loss).backward()
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                if self.truncate_grad:
                    clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1

        return {
            'actor_loss': total_actor_loss / max(update_count, 1),
            'value_loss': total_value_loss / max(update_count, 1),
            'entropy': total_entropy / max(update_count, 1),
        }

    def train(self):
        self.start_time = time.time()
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("collect rollout")
        self.time_report.add_timer("ppo update")
        self.time_report.start_timer("algorithm")

        self.env.begin_epoch(epoch=0, max_epochs=self.max_epochs)
        self._current_obs = self.env.reset()

        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()
            self.env.begin_epoch(epoch=epoch, max_epochs=self.max_epochs)

            # Learning rate schedule
            if self.lr_schedule == 'linear':
                frac = 1.0 - float(epoch) / float(self.max_epochs)
                actor_lr = self.actor_lr * frac
                critic_lr = self.critic_lr * frac
                for pg in self.actor_optimizer.param_groups:
                    pg['lr'] = max(actor_lr, 1e-6)
                for pg in self.critic_optimizer.param_groups:
                    pg['lr'] = max(critic_lr, 1e-6)

            # Collect rollout
            self.time_report.start_timer("collect rollout")
            self.collect_rollout()
            self.time_report.end_timer("collect rollout")

            # PPO update
            self.time_report.start_timer("ppo update")
            losses = self.update()
            self.time_report.end_timer("ppo update")

            self.iter_count += 1
            time_end_epoch = time.time()
            time_elapse = time.time() - self.start_time

            # Logging
            self.writer.add_scalar('actor_loss/iter', losses['actor_loss'], self.iter_count)
            self.writer.add_scalar('value_loss/iter', losses['value_loss'], self.iter_count)
            self.writer.add_scalar('entropy/iter', losses['entropy'], self.iter_count)

            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss

                self.writer.add_scalar('policy_loss/iter', mean_policy_loss, self.iter_count)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, self.iter_count)
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)
            else:
                mean_policy_loss = np.inf
                mean_episode_length = 0

            fps = self.steps_num * self.num_envs / max(time_end_epoch - time_start_epoch, 1e-6)
            print(
                'iter {}: ep loss {:.2f}, ep len {:.1f}, fps {:.0f}, '
                'actor_loss {:.4f}, value_loss {:.4f}, entropy {:.4f}'.format(
                    self.iter_count,
                    mean_policy_loss,
                    mean_episode_length,
                    fps,
                    losses['actor_loss'],
                    losses['value_loss'],
                    losses['entropy'],
                )
            )

            self.writer.flush()

            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + "_iter{}_reward{:.3f}".format(
                    self.iter_count, -mean_policy_loss
                ) if hasattr(self, 'name') else "policy_iter{}".format(self.iter_count))

        self.time_report.end_timer("algorithm")
        self.time_report.report()

        self.save('final_policy')

        np.save(os.path.join(self.log_dir, 'episode_loss_his.npy'), np.array(self.episode_loss_his))
        np.save(os.path.join(self.log_dir, 'episode_length_his.npy'), np.array(self.episode_length_his))

        self.close()

    def save(self, filename=None):
        if filename is None:
            filename = 'best_policy'
        # SHAC-compatible format: [actor, critic, target_critic, obs_rms, ret_rms]
        # PPO has no target critic — save a copy of critic in that slot.
        torch.save(
            [self.actor, self.critic, copy.deepcopy(self.critic), self.obs_rms, self.ret_rms],
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path, *, reset_optimizers=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.obs_rms = checkpoint[3].to(self.device) if checkpoint[3] is not None else None
        self.ret_rms = checkpoint[4].to(self.device) if checkpoint[4] is not None else None
        if reset_optimizers:
            self._build_optimizers()

    def close(self):
        self.writer.close()
