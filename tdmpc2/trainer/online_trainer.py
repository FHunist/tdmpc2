from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training.
	
	### RPG adjustments: support for batched (multi-environment) training ###
	author: @FHunist"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._num_envs = self.cfg.num_envs

		# Episode tracking
		self._env_episodes = [[] for _ in range(self._num_envs)]
		self._episode_rewards = torch.zeros(self._num_envs)
		self._episode_successes = [False] * self._num_envs
		self._episode_steps = torch.zeros(self._num_envs, dtype=torch.int)

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent. Both single/vectorized envs"""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			print("Current evaluation episode: ", i, " out of ", self.cfg.eval_episodes)
			obs, ep_reward, t = self.env.reset(), 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			for _ in range (10): #TODO: make this a parameter
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			### bootleg fix for dealing with vecEnv info - evaluation should later be replaced with flighmare evaluation"
			if isinstance(info, list):
				success = info[0].get('success', False) if isinstance(info[0], dict) else False
			else:
				success = info.get('success', False) if isinstance(info, dict) else False
				
			ep_rewards.append(ep_reward)
			ep_successes.append(success)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		print("Shape of ep_rewards: ", ep_rewards, " Shape of ep_successes: ", ep_successes)
		input("Evaluation done. Press enter to continue.")
		avg_reward = 0
		for tensor in ep_rewards:
			avg_reward += np.nanmean(tensor)
		print("Average reward: ", avg_reward)
		return dict(
			episode_reward=(avg_reward),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode step with vectorized data."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			# For vectorized, obs has shape [num_envs, obs_dim]
			if obs.ndim == 2 and obs.shape[0] == self._num_envs:
				obs = obs.cpu() # already in correct format
			else:
				# For single env, obs has shape [obs_dim]
				obs = obs.unsqueeze(0).cpu()
		if action is None:
			if self._num_envs > 1:
				action = torch.full((self._num_envs, self.env.action_space.shape[0]), float('nan'))
			else:
				action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			if self._num_envs > 1:
				reward = torch.full((self._num_envs,), float('nan'))
			else:
				reward = torch.tensor(float('nan'))
		# Create TensorDict with batch size = (1, num_envs) for vecEnvs
		if self._num_envs > 1:
			td = TensorDict(
				obs=obs,
				action=action,
				reward=reward,
			batch_size=(self._num_envs,))
		else:
			td = TensorDict(
				obs=obs,
				action=action.unsqueeze(0),
				reward=reward.unsqueeze(0),
			batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		print("Training TD-MPC2 agent...")
		train_metrics, eval_next = {}, False

		# initial reset
		obs = self.env.reset()
		current_td = self.to_td(obs)

		for env_idx in range(self._num_envs):
			self._env_episodes[env_idx] = [current_td]

		print("Starting training loop...")
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			if eval_next:
				eval_metrics = self.eval()
				print("Evaluation done.")
				eval_metrics.update(self.common_metrics())
				self.logger.log(eval_metrics, 'eval')
				eval_next = False

			# Collect experience
			if self._step > self.cfg.seed_steps:
				actions = self.agent.act(obs, t0=False)
			else:
				actions = self.env.rand_act()
				print("collecting random actions. Step: ", self._step, " from ", self.cfg.seed_steps)
			next_obs, rewards, dones, infos = self.env.step(actions)

			current_td = self.to_td(next_obs, actions, rewards)

			# Update metrics and check for episode completion
			for env_idx in range(self._num_envs):
                # Add reward to episode total
				self._episode_rewards[env_idx] += rewards[env_idx]
				self._episode_steps[env_idx] += 1
                
                # Check for success
				if isinstance(infos[env_idx], dict) and infos[env_idx].get('success', False):
					self._episode_successes[env_idx] = True
                
                # Add current transition to this environment's episode
				self._env_episodes[env_idx].append(current_td)
                
                # If this environment is done, finalize its episode
				if dones[env_idx]:
					self.add_to_buffer(env_idx)
                    
                    # Log metrics
					train_metrics.update({
                        'episode_reward': self._episode_rewards[env_idx].item(),
                        'episode_success': self._episode_successes[env_idx],
                        'episode_length': self._episode_steps[env_idx].item(),
                    })
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
                    
                    # Reset episode tracking for this environment
					self._episode_rewards[env_idx] = 0
					self._episode_successes[env_idx] = False
					self._episode_steps[env_idx] = 0
            
            # Update observation for next step
			obs = next_obs
            
            # Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
                    
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)
	def add_to_buffer(self, env_idx):
		"""Add a complete episode from a specific environment to the buffer."""
		if not self._env_episodes[env_idx]:
			return
			
		# Create a new list to hold tensordicts with just env_idx data
		env_specific_tds = []
		
		# Extract data only for env_idx from each timestep
		for td in self._env_episodes[env_idx]:
			env_specific_td = TensorDict({
				'obs': td['obs'][env_idx].unsqueeze(0),  # [1, 24]
				'action': td['action'][env_idx].unsqueeze(0),  # [1, 4]
				'reward': td['reward'][env_idx].unsqueeze(0),  # [1]
			}, batch_size=(1,))
			env_specific_tds.append(env_specific_td)
		
		# Concatenate the env-specific tensordicts along the time dimension
		if env_specific_tds:
			episode = torch.cat(env_specific_tds, dim=0)
			print(f"Extracted episode shapes - obs: {episode['obs'].shape}, "
				f"action: {episode['action'].shape}, reward: {episode['reward'].shape}")
			
			# Add to buffer
			self.buffer.add(episode)
			self._ep_idx += 1
		
		# Reset this environment's episode
		self._env_episodes[env_idx] = []
		print(f"Added episode from env {env_idx} to buffer. Total episodes: {self._ep_idx}")