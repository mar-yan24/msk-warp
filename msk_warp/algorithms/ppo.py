"""PPO (Proximal Policy Optimization) for MuJoCo Warp environments.

Standard PPO with GAE, clipped surrogate objective, and entropy bonus.
Uses the same environment and network interfaces as SHAC but without
differentiable physics — only forward simulation (no_grad=True).
"""

import os
import time
import copy

import numpy as np
import torch
from torch.distributions.normal import Normal
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
        env_kwargs = {k: v for k, v in env_cfg.items() if k != 'name'}
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

        # PPO hyperparameters
        self.gamma = cfg['params']['config'].get('gamma', 0.99)
        self.lam = cfg['params']['config'].get('lambda', 0.95)
        self.steps_num = cfg['params']['config']['steps_num']
        self.max_epochs = cfg['params']['config']['max_epochs']

        self.lr = float(cfg['params']['config']['learning_rate'])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')

        self.clip_ratio = cfg['params']['config'].get('clip_ratio', 0.2)
        self.entropy_coef = cfg['params']['config'].get('entropy_coef', 0.01)
        self.value_coef = cfg['params']['config'].get('value_coef', 0.5)
        self.ppo_epochs = cfg['params']['config'].get('ppo_epochs', 4)
        self.num_mini_batches = cfg['params']['config'].get('num_mini_batches', 8)
        self.max_grad_norm = cfg['params']['config'].get('max_grad_norm', 0.5)
        self.target_kl = cfg['params']['config'].get('target_kl', None)

        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)

        # Observation normalization
        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device)

        # Logging
        self.log_dir = cfg['params']['general']['logdir']
        os.makedirs(self.log_dir, exist_ok=True)
        save_cfg = copy.deepcopy(cfg)
        yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
        self.save_interval = cfg['params']['config'].get('save_interval', 500)
        self.name = cfg['params']['config'].get('name', 'ppo')

        # Create actor and critic
        actor_name = cfg['params']['network'].get('actor', 'ActorStochasticMLP')
        critic_name = cfg['params']['network'].get('critic', 'CriticMLP')
        self.actor = ACTOR_MAP[actor_name](
            self.num_obs, self.num_actions, cfg['params']['network'], device=self.device,
        )
        self.critic = CRITIC_MAP[critic_name](
            self.num_obs, cfg['params']['network'], device=self.device,
        )

        # Single optimizer for both actor and critic
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
            betas=cfg['params']['config'].get('betas', [0.9, 0.999]),
        )

        # Rollout buffers
        self.obs_buf = torch.zeros(
            (self.steps_num, self.num_envs, self.num_obs),
            dtype=torch.float32, device=self.device,
        )
        self.act_buf = torch.zeros(
            (self.steps_num, self.num_envs, self.num_actions),
            dtype=torch.float32, device=self.device,
        )
        self.logprob_buf = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.val_buf = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.done_buf = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        self.truncated_buf = torch.zeros(
            (self.steps_num, self.num_envs),
            dtype=torch.float32, device=self.device,
        )

        # GAE outputs
        self.advantages = torch.zeros_like(self.rew_buf)
        self.returns = torch.zeros_like(self.rew_buf)

        # Counters
        self.iter_count = 0
        self.step_count = 0

        # Episode tracking
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.best_episode_reward = -np.inf

        self.time_report = TimeReport()

        self.save('init_policy')

    def _normalize_obs(self, obs):
        if self.obs_rms is not None:
            return self.obs_rms.normalize(obs)
        return obs

    @torch.no_grad()
    def collect_rollout(self):
        """Collect steps_num steps of experience from the vectorized env."""
        obs = self.current_obs

        for t in range(self.steps_num):
            norm_obs = self._normalize_obs(obs)

            # Actor forward
            mu = self.actor.mu_net(norm_obs)
            std = self.actor.logstd.exp()
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

            # Critic forward
            value = self.critic(norm_obs).squeeze(-1)

            # Env step — actions are tanh'd by the env
            clipped_action = torch.tanh(action)
            next_obs, rew, done, extras, _, _ = self.env.step(clipped_action)

            # Distinguish termination from truncation
            terminated = extras.get('episode_end', torch.zeros_like(done))
            truncated = (done.bool() & ~terminated.bool()).float()

            # Store rollout data
            self.obs_buf[t] = obs
            self.act_buf[t] = action
            self.logprob_buf[t] = log_prob
            self.val_buf[t] = value
            self.rew_buf[t] = rew * self.rew_scale
            self.done_buf[t] = done.float()
            self.truncated_buf[t] = truncated

            # Update obs normalization
            if self.obs_rms is not None:
                self.obs_rms.update(next_obs)

            # Episode tracking
            self.episode_reward += rew
            self.episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                self.episode_reward_meter.update(self.episode_reward[done_env_ids])
                self.episode_length_meter.update(self.episode_length[done_env_ids].float())
                self.episode_reward[done_env_ids] = 0.0
                self.episode_length[done_env_ids] = 0

            obs = next_obs

        # Bootstrap value for the last observation
        norm_obs = self._normalize_obs(obs)
        self.bootstrap_value = self.critic(norm_obs).squeeze(-1)
        self.current_obs = obs
        self.step_count += self.steps_num * self.num_envs

    @torch.no_grad()
    def compute_gae(self):
        """Compute GAE-lambda advantages and returns."""
        lastgaelam = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                next_values = self.bootstrap_value
            else:
                next_values = self.val_buf[t + 1]

            # For terminated episodes, next value = 0.
            # For truncated episodes, keep bootstrapping (don't zero out).
            next_nonterminal = 1.0 - self.done_buf[t] + self.truncated_buf[t]

            delta = self.rew_buf[t] + self.gamma * next_values * next_nonterminal - self.val_buf[t]
            lastgaelam = delta + self.gamma * self.lam * next_nonterminal * lastgaelam
            self.advantages[t] = lastgaelam

        self.returns = self.advantages + self.val_buf

    def update(self):
        """Run PPO update epochs on the collected rollout."""
        # Flatten rollout buffers: (steps, envs, ...) -> (steps*envs, ...)
        total = self.steps_num * self.num_envs
        b_obs = self.obs_buf.reshape(total, -1)
        b_act = self.act_buf.reshape(total, -1)
        b_logprob = self.logprob_buf.reshape(total)
        b_adv = self.advantages.reshape(total)
        b_ret = self.returns.reshape(total)
        b_val = self.val_buf.reshape(total)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Normalize observations for network forward pass
        if self.obs_rms is not None:
            b_obs_norm = self.obs_rms.normalize(b_obs)
        else:
            b_obs_norm = b_obs

        batch_size = total // self.num_mini_batches
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        update_count = 0
        early_stopped = False

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(total, device=self.device)

            for start in range(0, total, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs_norm[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logprob = b_logprob[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]

                # Recompute log_prob, entropy, value under current policy
                mu = self.actor.mu_net(mb_obs)
                std = self.actor.logstd.exp()
                dist = Normal(mu, std)
                new_logprob = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1)
                new_value = self.critic(mb_obs).squeeze(-1)

                # PPO clipped surrogate
                log_ratio = new_logprob - mb_logprob
                ratio = log_ratio.exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((new_value - mb_ret) ** 2).mean()

                # Total loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1

                # Approximate KL for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    total_kl += approx_kl

            # Early stop if KL divergence exceeds target
            if self.target_kl is not None:
                avg_kl = total_kl / update_count
                if avg_kl > self.target_kl:
                    early_stopped = True
                    break

        self.actor_loss = total_actor_loss / update_count
        self.value_loss = total_value_loss / update_count
        self.mean_entropy = total_entropy / update_count
        self.mean_kl = total_kl / update_count
        self.ppo_epochs_run = epoch + 1 if early_stopped else self.ppo_epochs

    def train(self):
        self.start_time = time.time()

        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("collect rollout")
        self.time_report.add_timer("compute gae")
        self.time_report.add_timer("ppo update")

        self.time_report.start_timer("algorithm")

        # Initialize
        self.current_obs = self.env.reset()
        if self.obs_rms is not None:
            self.obs_rms.update(self.current_obs)

        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # Learning rate schedule
            if self.lr_schedule == 'linear':
                lr = (1e-5 - self.lr) * float(epoch / self.max_epochs) + self.lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = self.lr

            # Collect rollout
            self.time_report.start_timer("collect rollout")
            self.collect_rollout()
            self.time_report.end_timer("collect rollout")

            # Compute advantages
            self.time_report.start_timer("compute gae")
            self.compute_gae()
            self.time_report.end_timer("compute gae")

            # PPO update
            self.time_report.start_timer("ppo update")
            self.update()
            self.time_report.end_timer("ppo update")

            self.iter_count += 1
            time_end_epoch = time.time()

            # Logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar('lr/iter', lr, self.iter_count)
            self.writer.add_scalar('actor_loss/iter', self.actor_loss, self.iter_count)
            self.writer.add_scalar('value_loss/iter', self.value_loss, self.iter_count)
            self.writer.add_scalar('entropy/iter', self.mean_entropy, self.iter_count)
            self.writer.add_scalar('kl/iter', self.mean_kl, self.iter_count)
            self.writer.add_scalar('ppo_epochs_run/iter', self.ppo_epochs_run, self.iter_count)

            mean_episode_reward = -np.inf
            mean_episode_length = 0
            if self.episode_reward_meter.current_size > 0:
                mean_episode_reward = self.episode_reward_meter.get_mean()
                mean_episode_length = self.episode_length_meter.get_mean()

                if mean_episode_reward > self.best_episode_reward:
                    print_info("save best policy with reward {:.2f}".format(mean_episode_reward))
                    self.save()
                    self.best_episode_reward = mean_episode_reward

                self.writer.add_scalar('rewards/step', mean_episode_reward, self.step_count)
                self.writer.add_scalar('rewards/time', mean_episode_reward, time_elapse)
                self.writer.add_scalar('rewards/iter', mean_episode_reward, self.iter_count)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)

            fps = self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch)
            print(
                'iter {}: ep reward {:.2f}, ep len {:.1f}, '
                'fps {:.0f}, actor_loss {:.4f}, value_loss {:.4f}, '
                'entropy {:.4f}, kl {:.4f}'.format(
                    self.iter_count,
                    mean_episode_reward,
                    mean_episode_length,
                    fps,
                    self.actor_loss,
                    self.value_loss,
                    self.mean_entropy,
                    self.mean_kl,
                )
            )

            self.writer.flush()

            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + "_iter{}_reward{:.1f}".format(
                    self.iter_count, mean_episode_reward,
                ))

        self.time_report.end_timer("algorithm")
        self.time_report.report()

        self.save('final_policy')
        self.close()

    def save(self, filename=None):
        if filename is None:
            filename = 'best_policy'
        torch.save(
            [self.actor, self.critic, self.obs_rms],
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.obs_rms = checkpoint[2].to(self.device) if checkpoint[2] is not None else None

    def close(self):
        self.writer.close()
