from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .grf import Academy_3_vs_1_with_Keeper, Academy_Counterattack_Easy, Academy_Counterattack_Hard
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["academy_3_vs_1_with_keeper"]= partial(env_fn, env=Academy_3_vs_1_with_Keeper)
REGISTRY["academy_counterattack_easy"]= partial(env_fn, env=Academy_Counterattack_Easy)
REGISTRY["academy_counterattack_hard"]= partial(env_fn, env=Academy_Counterattack_Hard)

#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
