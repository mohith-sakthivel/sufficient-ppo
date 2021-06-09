from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution)

from stable_baselines3.common.preprocessing import maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from mre.models import MLP


class BaseModel(nn.Module, ABC):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    Parameters:
    ----------
    obs_space       - Observation space

    action_space            - Action space

    scale_imgs              - Whether to scale images by dividing with 255.0 (True by default)

    opt_cls                 - The optimizer to use. default:`optim.Adam`

    opt_kwargs              - Additional optimizer keyword arguments (excluding learning rate)
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        scale_imgs: bool = True,
        opt_cls: Type[optim.Optimizer] = optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = dict(),
    ):
        super(BaseModel, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.scale_imgs = scale_imgs

        self.opt_cls = opt_cls
        self.opt_kwargs = opt_kwargs
        self.optimizer = None  # type: Optional[optim.Optimizer]

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        Returns:
            The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            obs_space=self.obs_space,
            action_space=self.action_space,
            scale_imgs=self.scale_imgs,
        )

    @property
    def device(self) -> torch.device:
        """
        Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.
        """
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.
        """
        torch.save({"state_dict": self.state_dict(),
                    "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = "auto") -> "BaseModel":
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = torch.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        nn.utils.vector_to_parameters(torch.FloatTensor(
            vector).to(self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """
        return nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()


class ActorCriticPolicy(BaseModel):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction)

    Parameters:
    ----------
    obs_space       - Observation space

    action_space            - Action space

    lr_schedule             - Learning rate schedule (could be constant)

    act_fn                  - Activation function

    ortho_init              - Whether to use or not orthogonal initialization

    log_std_init            - Initial value for the log standard deviation

    squash_output           - Whether to squash the output using a tanh function

    feat_ext_cls            - Features extractor to use

    feat_ext_kwargs         - Keyword arguments to pass to the features extractor

    vf_feat_ext_cls         - Features extractor class for value network \\
                              `shared` -> uses the feature extractor of the policy network \\
                              `copy` -> duplicates the feature extractor of the policy network

    vf_feat_ext_kwargs      - Keyword arguments to pass to the features extractor of value network.\
                              Used only when `vf_feat_ext_cls` is not `shared` or `copy`

    scale_imgs              - Whether to scale images by dividing with 255.0 (True by default)

    opt_cls                 - The optimizer to use. default:`optim.Adam`

    opt_kwargs              - Additional optimizer keyword arguments (excluding learning rate)
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Dict[str, List[int]]] = None,
        act_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        squash_output: bool = False,
        feat_ext_cls: Type[BaseFeaturesExtractor] = None,
        feat_ext_kwargs: Optional[Dict[str, Any]] = dict(),
        vf_feat_ext_cls: Optional[Union[BaseFeaturesExtractor, str]] = 'shared',
        vf_feat_ext_kwargs: Optional[Dict[str, Any]] = dict(),
        scale_imgs: bool = True,
        opt_cls: Type[optim.Optimizer] = optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if feat_ext_cls is None:
            feat_ext_cls = (NatureCNN if len(obs_space.shape) == 3
                            else FlattenExtractor)

        if opt_kwargs is None:
            opt_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if opt_cls == optim.Adam:
                opt_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(obs_space,
                                                action_space,
                                                opt_cls=opt_cls,
                                                opt_kwargs=opt_kwargs)
        self.feat_ext_cls = feat_ext_cls
        self.feat_ext_kwargs = feat_ext_kwargs
        self.vf_feat_ext_cls = vf_feat_ext_cls
        self.vf_feat_ext_kwargs = vf_feat_ext_kwargs
        self._squash_output = squash_output

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if feat_ext_cls == FlattenExtractor:
                net_arch = dict(pi=[64, 64], vf=[64, 64])
            else:
                net_arch = dict(pi=[], vf=[])

        self.net_arch = net_arch
        self.act_fn = act_fn
        self.ortho_init = ortho_init

        self.feat_ext = feat_ext_cls(self.obs_space,
                                     **self.feat_ext_kwargs)
        self.feat_dim = self.feat_ext.features_dim

        # setup value feature extractor
        if vf_feat_ext_cls == 'shared':
            # share extractor with policy network
            self.vf_feat_ext = None
            self.vf_feat_dim = self.feat_dim
        elif vf_feat_ext_cls == 'copy':
            # use a copy of the policy network
            self.vf_feat_ext = feat_ext_cls(self.obs_space,
                                            **self.feat_ext_kwargs)
            self.vf_feat_dim = self.vf_feat_ext.features_dim
        elif isinstance(vf_feat_ext_cls, BaseFeaturesExtractor):
            # use a different feature extractor
            self.vf_feat_ext = vf_feat_ext_cls(self.obs_space,
                                               **vf_feat_ext_kwargs)
            self.vf_feat_dim = self.vf_feat_ext.features_dim
        else:
            raise ValueError

        self.scale_imgs = scale_imgs
        self.log_std_init = log_std_init

        # Action distribution
        if isinstance(action_space, gym.spaces.Box):
            assert len(action_space.shape) == 1, "Action space must be a vector"
            self.action_dist = DiagGaussianDistribution(
                np.prod(action_space.shape))
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_dist = CategoricalDistribution(action_space.n)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_dist = MultiCategoricalDistribution(action_space.nvec)
        elif isinstance(action_space, gym.spaces.MultiBinary):
            self.action_dist = BernoulliDistribution(action_space.n)
        else:
            raise NotImplementedError(
                "Probability distribution not implemented for given action space")

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        lr_schedule     - Learning rate schedule.\
                          lr_schedule(1) is the initial learning rate
        """
        self.pi_head = MLP(input_dim=self.feat_dim,
                           layer_dims=self.net_arch['pi'],
                           act_fn=self.act_fn).to(self.device)

        self.vf_head = MLP(input_dim=self.vf_feat_dim,
                           layer_dims=self.net_arch['vf'],
                           act_fn=self.act_fn).to(self.device)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.pi_head.out_dim, log_std_init=self.log_std_init)
        else:
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=self.pi_head.out_dim)

        self.value_net = nn.Linear(self.vf_head.out_dim, 1)
        # Init weights: use orthogonal initialization as done in openai baselines
        if self.ortho_init:
            module_gains = {
                self.feat_ext: np.sqrt(2),
                self.pi_head: np.sqrt(2),
                self.vf_head: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.opt_cls(
            self.parameters(), lr=lr_schedule(1), **self.opt_kwargs)

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """ (float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _get_latent(self, obs: torch.Tensor, pi: bool = True, vf: bool = False
                    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        obs     - Raw observation
        pi      - Calculate latents for policy
        vf      - Calculate latents for value function

        Returns:
            Tensors of requested latents. `None` for others.
        """
        # Preprocess the observation if needed
        assert self.feat_ext is not None, "No features extractor was set"
        processed_obs = preprocess_obs(obs, self.obs_space,
                                       normalize_images=self.scale_imgs)
        pi_latent = vf_latent = None
        if pi:
            feat = self.feat_ext(processed_obs)
            pi_latent = self.pi_head(feat)
        if vf:
            if self.vf_feat_ext is None:
                vf_feat = feat if pi else self.feat_ext(processed_obs)
            else:
                vf_feat = self.vf_feat_ext(processed_obs)
            vf_latent = self.vf_head(vf_feat)

        return pi_latent, vf_latent

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _ = self._get_latent(observation, pi=True)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution.get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. scaling images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        if isinstance(observation, dict):
            observation = ObsDictWrapper.convert_dict(observation)
        else:
            observation = np.array(observation)

        # Handle the different cases for images
        # as PyTorch use channel first format
        observation = maybe_transpose(observation, self.obs_space)

        vectorized_env = is_vectorized_observation(
            observation, self.obs_space)

        observation = observation.reshape((-1,) + self.obs_space.shape)

        observation = torch.as_tensor(observation).to(self.device)
        with torch.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions[0]

        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                act_fn=self.act_fn,
                log_std_init=self.log_std_init,
                squash_output=self.squash_output,
                # dummy lr schedule, not needed for loading policy alone
                lr_schedule=self._dummy_schedule,
                ortho_init=self.ortho_init,
                opt_cls=self.opt_cls,
                opt_kwargs=self.opt_kwargs,
                feat_ext_cls=self.feat_ext_cls,
                feat_ext_kwargs=self.feat_ext_kwargs,
                vf_feat_ext_cls=self.vf_feat_ext_cls,
                vf_feat_ext_kwargs=self.vf_feat_ext_kwargs
            )
        )
        return data

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self._get_latent(obs, pi=True, vf=True)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf = self._get_latent(obs, pi=True, vf=True)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
