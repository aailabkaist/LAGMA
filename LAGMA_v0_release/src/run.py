import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from modules.vae import REGISTRY as vae_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
import pickle
import numpy as np

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env,
                                     args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for iter in range(args.test_nepisode):
        runner.run(test_mode=True, t_episode=iter)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "goals": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "flag_win": {"vshape": (1,), "dtype": th.uint8}, # for monitoring
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
   
    # Give runner the scheme # need to input VQVAE's embeddings
    
    # Setup VQVAE
    state_dim = buffer.scheme["state"]["vshape"]
    if args.use_vqvae:
        vqvae = vae_REGISTRY[args.vae](state_dim, state_dim, args)
        if args.use_cuda:
            vqvae.cuda()
    else:
        vqvae = None
    
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, vqvae=vqvae)
    args.max_seq_length = buffer.max_seq_length
    # Learner   
    learner = le_REGISTRY[args.learner](mac, vqvae, buffer.scheme, logger, args)
    
    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = 0 # timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_vqvae_update_episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # update VQ-VAE once in a while ----------------------------------------
        if (args.vqvae_update_type==2) and (args.use_vqvae == True) and ((episode -  last_vqvae_update_episode) / (args.batch_size_run*args.vqvae_update_interval) >= 1.0):
            last_vqvae_update_episode = episode 
            if buffer.can_sample(args.vqvae_training_batch):
                emb_start_time = time.time()
                n_update = int(args.vqvae_training_batch / args.vqvae_training_mini_batch ) 

                for iter in range(0, n_update): # use mini-batch size 
                    batch_episode_sample = buffer.sample(args.vqvae_training_mini_batch)
                    # Truncate batch to only filled timesteps
                    max_ep_t = batch_episode_sample.max_t_filled()
                    batch_episode_sample = batch_episode_sample[:, :max_ep_t]

                    if batch_episode_sample.device != args.device:
                        batch_episode_sample.to(args.device)

                    learner.vqvae_train(batch_episode_sample, runner.t_env)

                emb_end_time = time.time()
                total_time = emb_end_time - emb_start_time

                if os.name != 'nt':
                    print("Processing time for memory embedding:", total_time )
        #-----------------------------------------------------------------------

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                test_batch = runner.run(test_mode=True) # not insert batch data

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            #save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            save_folder = args.config_name + '_' + args.env_args['map_name']
            save_path = os.path.join(args.local_results_path, "models", save_folder, args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)
            
            #.. replay buffer save
            replay_buffer_file_name = 'replay_buffer.pickle'
            win_flag_file_name = 'win_flag.pickle'
            file_path = os.path.join(save_path, replay_buffer_file_name)
            winflag_path = os.path.join(save_path, win_flag_file_name)

            cur_size = buffer.episodes_in_buffer
            sampled_replay_buffer = buffer.sample(cur_size)
            sampled_states = np.array(sampled_replay_buffer["state"])
            sampled_winflag = np.array(sampled_replay_buffer["flag_win"])

            with open(file_path, 'wb') as file:
                pickle.dump(sampled_states, file)
                
            with open(winflag_path, 'wb') as file:
                pickle.dump(sampled_winflag, file)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
