from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
<<<<<<< HEAD
from .grf import Academy_3_vs_1_with_Keeper, Academy_Counterattack_Easy, Academy_Counterattack_Hard
=======
>>>>>>> 257e39a5ad4ce051b1a491b5aa056c1a6cc15889
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
<<<<<<< HEAD
REGISTRY["academy_3_vs_1_with_keeper"]= partial(env_fn, env=Academy_3_vs_1_with_Keeper)
REGISTRY["academy_counterattack_easy"]= partial(env_fn, env=Academy_Counterattack_Easy)
REGISTRY["academy_counterattack_hard"]= partial(env_fn, env=Academy_Counterattack_Hard)
=======
>>>>>>> 257e39a5ad4ce051b1a491b5aa056c1a6cc15889

#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
