import os
import yaml
import time
import pathlib
import torch

from ppo.ppo import PPO
from ppo.policy import ActorCriticPolicy
from ppo.wrappers import atari_wrapper
from ppo.defaults import get_env_defaults

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

USE_CUSTOM = True


def start_experiment(args):
    # setup logging
    args.time_stamp = time.strftime("%d-%b-%y_%H:%M:%S")
    path_str = '_'.join([args.exp_name, 's{}'.format(args.seed), args.time_stamp])
    logdir = pathlib.Path(args.logdir) / path_str
    os.makedirs(logdir)
    # save experiment args
    with open(logdir.joinpath('params.yml'), 'w') as out:
        yaml.dump(args.__getstate__(), out, indent=4)
    # setup environments
    env = make_env(args.env_name, args.workers, args.seed, **args.env)

    if USE_CUSTOM:
        # use custom PPO
        model = PPO(policy=ActorCriticPolicy,
                    env=env,
                    tb_log=logdir,
                    create_eval_env=False,
                    verbose=1,
                    seed=args.seed,
                    **args.algo)
    else:
        # use stablebaselines-3 PPO
        from stable_baselines3.ppo.ppo import PPO as PPOv0

        model = PPOv0(policy='CnnPolicy',
                      env=env,
                      learning_rate=args.algo.learning_rate,
                      n_steps=args.algo.n_steps,
                      batch_size=args.algo.batch_size,
                      n_epochs=args.algo.n_epochs,
                      clip_range=args.algo.clip_range,
                      ent_coef=args.algo.ent_coef,
                      tensorboard_log=logdir,
                      verbose=1,
                      seed=args.seed,
                      device=args.algo.device)

    model.learn(total_timesteps=args.timesteps,
                log_interval=args.log_interval,
                eval_env=None,
                eval_freq=-1,
                n_eval_episodes=5)

    model.save(logdir / args.exp_name)

    del model  # remove to demonstrate saving and loading
    env = make_env(args.env_name, args.workers, args.seed, **args.env)
    model = PPO.load(path=logdir / args.exp_name, env = env)

    obs = env.reset()
    for _ in range(1):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    print('Done....')


def make_env(env_name, n_envs, seed=None, env_type=None,
             monitor_dir=None, env_kwargs=None, multi_process=False,
             wrapper_kwargs={}):
    """
    Make environment based with necessary wrappers
    """
    wrapper_kwargs = wrapper_kwargs.copy()
    vec_env_cls = (SubprocVecEnv if (multi_process and n_envs > 1)
                   else DummyVecEnv)
    frame_stack = wrapper_kwargs.pop('frame_stack', None)
    # choose wrappers based on env type
    if env_type == 'atari':
        def wrapper_func(x): return atari_wrapper(x, **wrapper_kwargs)
    else:
        wrapper_func = None
    # make vectorized env
    env = make_vec_env(env_id=env_name,
                       n_envs=n_envs,
                       seed=seed,
                       monitor_dir=monitor_dir,
                       wrapper_class=wrapper_func,
                       env_kwargs=env_kwargs,
                       vec_env_cls=vec_env_cls)
    if frame_stack is not None:
        env = VecFrameStack(env, frame_stack)
    return env


def get_default_args(env_type):
    args = get_env_defaults(env_type)
    args.seed = 0
    args.log_interval = 1
    # Environment specific args
    args.env.multi_process = True
    # Algorithm specific args
    args.algo.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment args
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--timesteps', type=int, default=1e7)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    # environment args
    parser.add_argument('--env_type', type=str, default='atari')
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')

    cli_args = parser.parse_args()
    algo_args = []
    args = get_default_args(cli_args.env_type)

    for key, val in cli_args.__dict__.items():
        if val is not None:
            if key in algo_args:
                args.algo[key] = val
            else:
                args[key] = val

    start_experiment(args)
