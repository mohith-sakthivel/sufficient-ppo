import time
import warnings
from typing import Any, Dict, Optional, Type, Union, Tuple, List

import gym
import numpy as np
import torch as torch
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import safe_mean

from ppo.base_class import BaseAlgorithm


def constant_fn(val):
    """ Create a function that returns a constant """
    def func(_):
        return val
    return func


class PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    Note: (RT) - parameters with this tag can be a function of remaining training (from 1 to 0)

    Parameters
    -----------

    policy          - Policy network class

    env             - Environment to learn from

    lr              - Learning rate for policy and value networks (RT)

    n_steps         - The number of steps to run for each environment per update \
                      (i.e. rollout buffer size is n_steps * n_envs)

    batch_size      - Minibatch size for each training epoch

    n_epochs        - Number of epoch when optimizing the surrogate loss

    gamma           - Discount factor

    gae_lambda      - Factor for trading-off of bias vs variance in GAE calculation

    clip_range      - Clipping parameter (RT)

    clip_range_vf   - Clipping parameter for the value function (RT) \
                      Parameter specific to the OpenAI implementation. \
                      `Note`: This clipping depends on the reward scaling. \
                      Default: None (No clipping)

    ent_coef        - Entropy coefficient for the loss calculation

    vf_coef         - Value function coefficient for the loss calculation

    max_grad_norm   - Maximum value for the gradient clipping

    target_kl       - Limit the KL divergence between updates (default: no limit)

    tb_log          - Log location for tensorboard (if None, no logging)


    create_eval_env - Whether to create a second environment that will be used for 
                      evaluating the agent periodically.
                      (Only available when passing string for the environment)

    pol_kwargs      - Additional arguments to be passed to the policy on creation

    verbose         - Verbosity level: 0 no output, 1 info, 2 debug

    seed            - Seed for the pseudo random generators

    device          - Device (cpu, cuda, auto, ...)
    """

    def __init__(
        self,
        policy: ActorCriticPolicy,
        env: Union[GymEnv, str],
        lr: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        tb_log: Optional[str] = None,
        create_eval_env: bool = False,
        pol_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
    ):
        super(PPO, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            lr=lr,
            pol_kwargs=pol_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tb_log=tb_log,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.lr_schedule = self.lr if callable(
            self.lr) else constant_fn(self.lr)
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        assert (batch_size > 1), "`batch size` must be > 1. Otherwise, will lead to \
                                  noisy gradient and NaN because of the advantage normalization"
        if clip_range_vf is not None:
            if isinstance(clip_range_vf, (float, int)):
                assert clip_range_vf > 0, "`clip_range_vf` must be positive or \
                                           None should be used for no clipping"
                clip_range_vf = constant_fn(clip_range_vf)

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN during advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (buffer_size > 1), "`n_steps * n_envs` must be greater than 1"
            if buffer_size % batch_size > 0:
                warnings.warn(
                    "The specified buffer size is not divisible by the mini-batch size")

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range if callable(
            clip_range) else constant_fn(clip_range)
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        self._setup_model()

    def _setup_model(self):
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps,
                                            self.observation_space,
                                            self.action_space,
                                            self.device,
                                            gamma=self.gamma,
                                            gae_lambda=self.gae_lambda,
                                            n_envs=self.n_envs)

        self.policy = self.policy_class(self.observation_space,
                                        self.action_space,
                                        self.lr_schedule,
                                        **self.pol_kwargs)
        self.policy = self.policy.to(self.device)

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``
        by interacting with the environment

        Parameters
        -----------
        env             - Training environment

        callback        - Callback that will be called at each step \
                          and at the beginning and end of the rollout

        rollout_buffer  - Buffer to fill with rollouts

        n_steps         - Number of experiences to collect per environment

        Return
        -------
            True if function returned with at least `n_rollout_steps` collected
            False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:

            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards,
                               self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = torch.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor, pi=False, vf=True)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_lr(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._remain_progress)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._remain_progress)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / \
                    (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * \
                    torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(values, rollout_data.returns, reduction='none')
                if self.clip_range_vf is not None:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                    value_loss_clipped = F.mse_loss(values_pred, rollout_data.returns, reduction='none')
                    value_loss = torch.maximum(value_loss, value_loss_clipped)

                value_loss = 0.5 * value_loss.mean()
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates,
                      exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "ppo",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True):
        """
        Parameters
        -----------
        total_timesteps      - Total timesteps
        callback             - Can be a callable, callback, list of callbacks or None. \
                               Called at every step with state of the algorithm.
        log_interval         - Frequency at which to log
        eval_env             - Environment for evaluation (optional)
        eval_freq            - Frequency of evaluation (optional)
        n_eval_episodes      - Number of evaluation episodes
        tb_log_name          - Tensorboard logger name
        eval_log_path        - Evaluation logging path (optional)
        reset_num_timesteps  - Whether to reset or not the ``num_timesteps`` attribute
        """

        iteration = 0

        total_timesteps, callback = self._setup_learn(total_timesteps=total_timesteps,
                                                      eval_env=eval_env,
                                                      callback=callback,
                                                      eval_freq=eval_freq,
                                                      n_eval_episodes=n_eval_episodes,
                                                      log_path=eval_log_path,
                                                      reset_num_timesteps=reset_num_timesteps,
                                                      tb_log_name=tb_log_name)

        callback.on_training_start(locals(), globals())
        last_log_timestep = self.num_timesteps
        last_log_time = time.time()
        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_remain_progress(
                self.num_timesteps, total_timesteps)
            
            self.train()

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - last_log_timestep)/ (time.time() - last_log_time))
                logger.record("time/iterations", iteration,
                              exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record(
                        "rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record(
                        "rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() -
                              self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps",
                              self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

        callback.on_training_end()

    def _get_torch_save_params(self):
        """
        Get the name of the torch variables that will be saved with ``torch.save``,
        ``torch.load`` and ``state_dicts`` instead of the default pickling strategy.
        This is to handle device placement correctly.

        Names can point to specific variables under classes
        e.g. "policy.optimizer" would point to the ``optimizer`` object of ``self.policy``

        Return
        -------
            List of Torch variables whose state dicts to save (e.g. nn.Modules),
            and list of other Torch variables to store with ``torch.save``.
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
