import gym
import os

custom_envs = {
            # Contextual environments
            "CartPoleContextual-v0":
                dict(path='gym_extensions.discrete.classic.cartpole_contextual:CartPoleContextualEnv',
                     max_episode_steps=500,
                     kwargs= dict()),
            "PendulumContextual-v0":
                dict(path='gym_extensions.discrete.classic.pendulum_contextual:PendulumContextualEnv',
                     max_episode_steps=500,
                     kwargs= dict())
                     }

def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key,
                        entry_point=value["path"],
                        max_episode_steps=value["max_episode_steps"],
                        kwargs=value["kwargs"])

        if "reward_threshold" in value:
            arg_dict["reward_threshold"] = value["reward_threshold"]

        gym.envs.register(**arg_dict)

register_custom_envs()
