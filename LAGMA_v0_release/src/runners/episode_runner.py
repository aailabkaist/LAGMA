from re import I
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import copy
import scipy.io as sio
from scipy.io import savemat
import os
import logging

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, vqvae=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        if vqvae is not None:
            self.vqvae = vqvae

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, t_episode=0):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        if self.args.verbose == True:
            trj_save_path = os.path.join(self.args.local_results_path,
                                "pic_trajectories",
                                self.args.trajectory_saveStr)

            if t_episode == 0:
                if not os.path.exists(trj_save_path):
                    os.makedirs(trj_save_path)

            buf_Timestep = []
            buf_action   = []            
            buf_state    = []            
            buf_winflag  = []
            buf_visit_nodes =[]

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # Here actions: q-value, logp: goal-selected logp
            if self.args.agent == "lagma_gc":
                actions, goals = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, goal_latent=self.vqvae.emb.weight,test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            
            if self.args.verbose:
                action_detach  = actions.detach().cpu().squeeze().numpy()
                state_detach   = self.batch['state'][0][self.t].detach().cpu().squeeze().numpy()
                winflag_detach = self.batch['flag_win'][0][self.t].detach().cpu().squeeze().numpy()
                state_input = th.tensor(state_detach).to(self.args.device).unsqueeze(0)
                recon_s, z_e, latent_emb, visit_node = self.vqvae(state_input)
                visit_node = visit_node.cpu().numpy()

                #.. save trajectory information
                if self.args.saveTrj == True :                    
                    buf_Timestep.append(copy.deepcopy(self.t))            # for all agents
                    buf_action.append(copy.deepcopy(action_detach))       # for all agents
                    buf_state.append(copy.deepcopy(state_detach))         # for all agents                    
                    buf_winflag.append(copy.deepcopy(winflag_detach))     # for all agents
                    buf_visit_nodes.append(copy.deepcopy(visit_node))     # for all agents
            

            if 'academy' in self.args.env: 
                if self.args.agent == "lagma_gc":
                    post_transition_data = {
                        "actions": cpu_actions,
                        "goals": goals,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                        "flag_win": [(bool(env_info['score_reward']),)], 
                    }
                else:
                    post_transition_data = {
                        "actions": cpu_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                        "flag_win": [(bool(env_info['score_reward']),)], 
                    }
            else:
                if self.args.agent == "lagma_gc":
                    post_transition_data = {
                        "actions": cpu_actions,
                        "goals": goals,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                        "flag_win": [(bool(env_info['battle_won']),)],
                    }
                else:
                    post_transition_data = {
                        "actions": cpu_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                        "flag_win": [(bool(env_info['battle_won']),)],
                    }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        if self.args.verbose:
            #.. save trajectory information
            if self.args.saveTrj == True : 
                # append last data
                action_detach  = actions.detach().cpu().squeeze().numpy()
                state_detach   = self.batch['state'][0][self.t-1].detach().cpu().squeeze().numpy()
                winflag_detach = self.batch['flag_win'][0][self.t-1].detach().cpu().squeeze().numpy()
                state_input = th.tensor(state_detach).to(self.args.device).unsqueeze(0)
                recon_s, z_e, latent_emb, visit_node = self.vqvae(state_input)
                visit_node = visit_node.cpu().numpy()

                buf_Timestep.append(copy.deepcopy(self.t-1))          # for all agents
                buf_action.append(copy.deepcopy(action_detach))       # for all agents
                buf_state.append(copy.deepcopy(state_detach))         # for all agents                    
                buf_winflag.append(copy.deepcopy(winflag_detach))     # for all agents     
                buf_visit_nodes.append(copy.deepcopy(visit_node))     # for all agents

                mdic = {"timestep":      np.array( buf_Timestep),                         
                        "ally_action":   np.array( buf_action  ),                        
                        "state"       :  np.array( buf_state   ),
                        "win_flag"    :  np.array( buf_winflag ),
                        "visit_node"  :  np.array( buf_visit_nodes ),
                    "label":"LAGMA"}
                #savemat( saveStr , mdic )
                saveStr_mat = self.args.trajectory_saveStr + "_" + str(t_episode) + ".mat"    
                sio.savemat( os.path.join(trj_save_path, saveStr_mat), mdic )   
                
                saveStr_s_traj = "s_traj_" + str(t_episode)
                np.save(os.path.join(trj_save_path, saveStr_s_traj), buf_state )           


        # Select actions in the last stored state
        if self.args.agent == "lagma_gc":
            actions, goals = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, goal_latent=self.vqvae.emb.weight.detach(),test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            self.batch.update({"actions": cpu_actions, "goals": goals }, ts=self.t)
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            self.batch.update({"actions": cpu_actions }, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
