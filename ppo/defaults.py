from ppo.utils import AttrDict, ProgressSchedule


def default_args_template() -> AttrDict:
    args = AttrDict()
    args.env = AttrDict()
    args.algo = AttrDict()
    args.algo.pol_kwargs = AttrDict()
    return args


def mujoco_defaults() -> AttrDict:
    args = default_args_template()
    args.env.env_type = 'mujoco'
    args.workers = 1  # baselines/PPO paper
    args.algo.n_steps = 2048
    args.algo.batch_size = 64
    args.algo.gae_lambda = 0.95
    args.algo.gamma = 0.99
    args.algo.n_epochs = 10
    args.algo.ent_coef = 0
    args.algo.lr = ProgressSchedule(3e-4)
    # baselines uses a single variable for both policy and vf clipping
    args.algo.clip_range = 0.2
    args.algo.clip_range_vf = 0.2
    args.algo.pol.value_features_extractor_class = 'copy'
    return args


def atari_defaults() -> AttrDict:
    args = default_args_template()
    args.env.env_type = 'atari'
    args.env.wrapper_kwargs = {'frame_stack': 4}
    args.workers = 8  # PPO paper
    args.algo.n_steps = 128
    args.algo.batch_size = 256
    args.algo.gae_lambda = 0.95
    args.algo.gamma = 0.99
    args.algo.n_epochs = 4
    args.algo.ent_coef = 0.01
    args.algo.lr = ProgressSchedule(2.5e-4)
    # baselines uses a single variable for both policy and vf clipping
    args.algo.clip_range = 0.1
    args.algo.clip_range_vf = 0.1
    return args


def get_env_defaults(env_type: str) -> AttrDict:
    if env_type == 'atari':
        return atari_defaults()
    elif env_type == 'mujoco':
        return mujoco_defaults()
    elif env_type == 'none':
        return default_args_template()
    else:
        raise NotImplementedError
