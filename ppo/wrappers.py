from stable_baselines3.common.atari_wrappers import AtariWrapper

def atari_wrapper(env, **kwargs):
    env = AtariWrapper(env, **kwargs)
    return env
