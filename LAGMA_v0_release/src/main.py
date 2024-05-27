import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run
import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

#ex = Experiment("pymarl")
ex = Experiment("pymarl", save_git_info=False) # modified
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config_env(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def _get_config_alg(params, arg_name, subfolder, map_name, task_name):
    config_name = None
    
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name_default = _v.split("=")[1]
            del params[_i]
            break

    # use task dependent configuration        
    if "sparse" in task_name: # sparse smac
        if map_name=="3m":
            config_name="lagma_sc2_sparse_3m"
        elif map_name=="8m":
            config_name="lagma_sc2_sparse_8m"
        elif map_name=="2s3z":
            config_name="lagma_sc2_sparse_2s3z"
        elif map_name=="2m_vs_1z":
            config_name="lagma_sc2_sparse_2m_vs_1z"
        else:
            config_name="lagma_sc2_sparse_3m"
            
    else: # dense smac
<<<<<<< HEAD
        
        #.. check GRF
        if "academy" in map_name:
            if "3_vs_1" in map_name:
                config_name="lagma_grf_3_vs_1WK"
            else:
                config_name="lagma_grf_CA_easy"
        else:
            #.. default: smac
            if map_name=="6h_vs_8z":
                config_name="lagma_sc2_6h_vs_8z"
            else:
                config_name="lagma_sc2"    
=======
        if map_name=="6h_vs_8z":
            config_name="lagma_sc2_6h_vs_8z"
        else:
            config_name="lagma_sc2"        
>>>>>>> 257e39a5ad4ce051b1a491b5aa056c1a6cc15889
                
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        #return config_dict
        return config_dict, config_name

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':

    params = deepcopy(sys.argv)
    warnings.filterwarnings("ignore")
    # Get the defaults from default.yaml
    #with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r", encoding="utf8", errors="ignore") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config= _get_config_env(params, "--env-config", "envs")
    config_dict = recursive_dict_update(config_dict, env_config)
        
    map_name=env_config['env_args']['map_name']
<<<<<<< HEAD
    if "academy" in map_name:
        task_name ="grf"
    else:
        dense_flag=env_config['env_args']['reward_only_positive']
        if dense_flag==False:
            task_name = "sc2_sparse"
        else:
            task_name="sc2"
=======
    dense_flag=env_config['env_args']['reward_only_positive']
    if dense_flag==False:
        task_name = "sc2_sparse"
    else:
        task_name="sc2"
>>>>>>> 257e39a5ad4ce051b1a491b5aa056c1a6cc15889

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "env_args.map_name":
            map_name = _v.split("=")[1]
            
    print("Task_name   >>>>> ",task_name)    
    print("Map_name    >>>>> ",map_name)
    alg_config, config_name = _get_config_alg(params, "--config", "algs", map_name,task_name)
    
    print("Config_file >>>>> ",config_name)
    config_dict = recursive_dict_update(config_dict, alg_config)
    
    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")

    if config_dict['config_name'] == '':
        cur_config_name = config_dict['agent']
    else:
        cur_config_name = config_dict['config_name']
    if config_dict['env_args']['map_name'] == '':
        save_folder = cur_config_name 
    else:
        save_folder = cur_config_name + '_' + config_dict['env_args']['map_name']

    save_folder   = cur_config_name
    file_obs_path = os.path.join(file_obs_path, save_folder )
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

