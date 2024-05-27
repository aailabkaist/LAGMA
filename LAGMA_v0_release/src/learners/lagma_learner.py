import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.vae import REGISTRY as vae_REGISTRY
import torch as th
from torch.optim import RMSprop, Adam
import time 
import numpy as np

class LAGMALearner:
    def __init__(self, mac, vqvae, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.device = self.args.device

        self.state_dim = scheme["state"]["vshape"]
        #input_shape    = self._get_input_shape(scheme) # input_shape includes agent-id (if available)
        
        self.params = list(mac.parameters())
        if self.args.use_vqvae:
            #self.vae = vae_REGISTRY[self.args.vae](self.state_dim, self.state_dim, self.args)
            self.vae = vqvae
            self.vae_params = list(self.vae.parameters())
            self.vae_optimizer = Adam(params=self.vae_params, lr=args.lr)
        
        self.update_vqvae = False
        self.update_codebook = False
        self.last_vqvae_update_episode = 0
        self.last_codebook_update_episode = 0
        self.last_target_update_episode = 0
        
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.vae_log_t = self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.vae_losses      = th.tensor(0.0).to(self.args.device)   
        self.ce_losses       = th.tensor(0.0).to(self.args.device)   
        self.vq_losses       = th.tensor(0.0).to(self.args.device)   
        self.commit_losses   = th.tensor(0.0).to(self.args.device)   
        self.coverage_losses = th.tensor(0.0).to(self.args.device)   

    def vqvae_train(self, batch: EpisodeBatch , t_env ):
        # Get the relevant quantities        
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # backward reward and reward sum generation for codebook update                
        rewards_th = th.tensor(batch["reward"]).to(self.device).squeeze(-1)
        sum_rewards = th.sum(th.tensor( rewards_th ).to(self.device), axis=1) # [bs]
        reward_tgo  = th.zeros_like(rewards_th ).to(self.device)

        ##.. reverse sequence for reward-to-go computation
        for t in range(batch.max_seq_length-1, -1, -1):
            if t == batch.max_seq_length-1:
                reward_tgo[:, t] = rewards_th[:,t]
            else:
                reward_tgo[:, t] = rewards_th[:,t] + self.args.gamma*reward_tgo[:, t+1]

        vae_losses      = th.tensor(0.0).to(self.args.device)
        #.. for monitoring
        ce_losses       = th.tensor(0.0).to(self.args.device)
        vq_losses       = th.tensor(0.0).to(self.args.device)
        commit_losses   = th.tensor(0.0).to(self.args.device)
        coverage_losses = th.tensor(0.0).to(self.args.device)

        # vqvae training ===============================================================
        if self.args.use_vqvae:
            for t in range(batch.max_seq_length):                
                #.. add vq-vae estimation -----
               
                state_input = th.tensor(batch["state"][:, t]).to(self.device)
                
                if self.args.recon_type==3:
                    timestep = th.tensor( [float(t) / float(self.args.max_seq_length)] ).repeat(self.args.batch_size).unsqueeze(-1).to(self.args.device)
                    embed_input = th.cat( [state_input, timestep], dim=1) # [bs,dim]
                    recon, z_e, latent_emb, argmin, Cqt_hat = self.vae(embed_input, timestep = timestep)
                else:
                    recon, z_e, latent_emb, argmin = self.vae(state_input)
                    
                if self.args.recon_type == 2:
                        Cqt_target = self.vae.call_Cqt_batch(argmin)                
                elif self.args.recon_type == 3:
                    if self.args.return_type == 1:
                        Cqt_target = self.vae.call_Cqt_batch(argmin)                
                    else:
                        Cqt_target = reward_tgo[:, t] # too varying --> high variance

                # codebook is not updated in vqvae training
                #self.vae.codebook_update(argmin, t_env, sum_rewards, reward_tgo[:,t]) # include for-batch

                #.. compute timedependent indexing ----------
                if t == 0:
                    if self.args.ref_max_time == 1:
                        dn = int(self.args.n_codes / batch.max_seq_length) # dn
                        dr = self.args.n_codes % batch.max_seq_length
                        ids = dn*batch.max_seq_length
                    elif self.args.ref_max_time == 2:
                        dn = int(self.args.n_codes / self.args.max_seq_length ) # dn
                        dr = self.args.n_codes % self.args.max_seq_length
                        ids = dn * self.args.max_seq_length
                        
                if dn >= 1:    
                    ndx = np.arange(dn*t, dn*(t+1), 1)      
                    if t < dr:
                        ndx = np.append(ndx, np.array(ids+t))
                       
                else:
                    ndx = np.array([int(t*dn)])
                        
                if self.args.timestep_emb == False:
                    ndx = None
                #---------------------------------------------

                if self.args.recon_type ==1:
                    vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                        self.vae.loss_function(state_input, recon, z_e, latent_emb, ndx=ndx)
                elif self.args.recon_type ==2:
                    vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                        self.vae.loss_function(Cqt_target, recon, z_e, latent_emb, ndx=ndx)
                elif self.args.recon_type ==3:
                    vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                        self.vae.loss_function(state_input, recon, z_e, latent_emb, ndx=ndx, Cqt=Cqt_target, recon_Cqt=Cqt_hat)                        

                vae_losses      += vae_loss
                ce_losses       += ce_loss
                vq_losses       += vq_loss
                commit_losses   += commit_loss
                coverage_losses += coverage_loss
        #==============================================================================

        if self.args.use_vqvae: 
            vae_losses      /= batch.batch_size 
            ce_losses       /= batch.batch_size 
            vq_losses       /= batch.batch_size 
            commit_losses   /= batch.batch_size 
            coverage_losses /= batch.batch_size 
            
            self.vae_losses      = vae_losses     
            self.ce_losses       = ce_losses      
            self.vq_losses       = vq_losses      
            self.commit_losses   = commit_losses  
            self.coverage_losses = coverage_losses

            self.vae_optimizer.zero_grad()
            grad_norm = th.nn.utils.clip_grad_norm_(self.vae_params, self.args.grad_norm_clip)
            vae_losses.backward()
            self.vae_optimizer.step()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities        
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        win_flag = batch["flag_win"].squeeze(-1)

        flag_des_trj = th.any( win_flag != 0, dim=1 ) # [bs]
        
        if th.any(flag_des_trj != 0 ):
            find_desirable_trj = 1

        # backward reward and reward sum generation for codebook update                
        rewards_th = th.tensor(batch["reward"]).to(self.device).squeeze(-1)
        sum_rewards = th.sum(th.tensor( rewards_th ).to(self.device), axis=1) # [bs]
        reward_tgo  = th.zeros_like(rewards_th ).to(self.device)

        #.. reverse sequence for reward-to-go computation
        for t in range(batch.max_seq_length-1, -1, -1):
            if t == batch.max_seq_length-1:
                reward_tgo[:, t] = rewards_th[:,t]
            else:
                reward_tgo[:, t] = rewards_th[:,t] + self.args.gamma*reward_tgo[:, t+1]
        
        if self.args.vqvae_update_type == 1: # training with the current batch from replay buffer
            if ((episode_num - self.last_vqvae_update_episode) / self.args.vqvae_update_interval >= 1.0):
                self.update_vqvae = True
                self.last_vqvae_update_episode = episode_num

                if (t_env >= self.args.vqvae_training_stop):
                    self.update_vqvae = False

            else:
                self.update_vqvae = False       
                if t_env <= self.args.buffer_update_time: # update vqvae at early trainig time
                    self.update_vqvae = True

            if ((episode_num - self.last_codebook_update_episode) / self.args.codebook_update_interval >= 1.0):
                self.update_codebook = True
                self.last_codebook_update_episode = episode_num
            else:
                self.update_codebook = False       
                if t_env <= self.args.buffer_update_time: # update codebook at early trainig time
                    self.update_codebook = True

        else:
            self.update_vqvae = False       

            if ((episode_num - self.last_codebook_update_episode) / self.args.codebook_update_interval >= 1.0):
                self.update_codebook = True
                self.last_codebook_update_episode = episode_num
            else:
                self.update_codebook = False       
                if t_env <= self.args.buffer_update_time: # update codebook at early trainig time
                    self.update_codebook = True
                
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        #vae_losses      = th.tensor(0.0).to(self.args.device)        
        #.. for monitoring
        ce_losses       = th.tensor(0.0).to(self.args.device)
        vq_losses       = th.tensor(0.0).to(self.args.device)
        commit_losses   = th.tensor(0.0).to(self.args.device)
        coverage_losses = th.tensor(0.0).to(self.args.device)
        
        visit_nodes =[]
        if self.args.flag_batch_wise_vqvae_loss:
            vae_losses  = [] # need masking
        else:
            vae_losses  = th.tensor(0.0).to(self.args.device)
        
        #ts = time.time()
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        
            #.. add vq-vae estimation -----
            if self.args.use_vqvae:
                state_input = th.tensor(batch["state"][:, t]).to(self.device)
                
            #if (self.args.flag_zero_state_management==False) or (self.args.flag_zero_state_management and sum(state_input)!=0):
                if self.args.recon_type==3:
                    timestep = th.tensor( [float(t) / float(self.args.max_seq_length)] ).repeat(self.args.batch_size).unsqueeze(-1).to(self.args.device)
                    embed_input = th.cat( [state_input, timestep], dim=1) # [bs,dim]
                    recon, z_e, latent_emb, argmin, Cqt_hat = self.vae(embed_input, timestep = timestep)
                else:
                    recon, z_e, latent_emb, argmin = self.vae(state_input)
                    
                if self.args.recon_type == 2:
                        Cqt_target = self.vae.call_Cqt_batch(argmin)                
                elif self.args.recon_type == 3:
                    if self.args.return_type == 1:
                        Cqt_target = self.vae.call_Cqt_batch(argmin)                
                    else:
                        Cqt_target = reward_tgo[:, t] # too varying --> high variance
                    
                # manage zero state vector
                if self.args.flag_zero_state_management==True:
                    sums = th.sum(state_input,dim=1)
                    #zero_index = th.nonzero(sums==0, as_tuple=False).squeeze()
                    zero_index = th.nonzero(sums==0, as_tuple=False)
                    if len(zero_index) > 0 :
                        argmin[zero_index] = self.args.n_codes
                        
                visit_nodes.append(argmin)
                                
            #.. update codebook (code book is updated after flag_buffer_update = True)
                if self.update_codebook:
                    self.vae.codebook_update(argmin, t_env, sum_rewards, reward_tgo[:,t]) # include for-batch

            # vqvae training ===============================================================
                if self.update_vqvae:
                    #.. compute timedependent indexing ----------
                    if t == 0:
                        if self.args.ref_max_time == 1:
<<<<<<< HEAD
                            dn   = int(self.args.n_codes / batch.max_seq_length) # dn
                            dn_r = (self.args.n_codes / batch.max_seq_length) # dn_r
                            dr   = self.args.n_codes % batch.max_seq_length
                            ids  = dn*batch.max_seq_length
                        elif self.args.ref_max_time == 2:
                            dn   = int(self.args.n_codes / self.args.max_seq_length ) # dn
                            dn_r = int(self.args.n_codes / self.args.max_seq_length ) # dn_r
                            dr   = self.args.n_codes % self.args.max_seq_length
                            ids  = dn * self.args.max_seq_length
=======
                            dn = int(self.args.n_codes / batch.max_seq_length) # dn
                            dr = self.args.n_codes % batch.max_seq_length
                            ids = dn*batch.max_seq_length
                        elif self.args.ref_max_time == 2:
                            dn = int(self.args.n_codes / self.args.max_seq_length ) # dn
                            dr = self.args.n_codes % self.args.max_seq_length
                            ids = dn * self.args.max_seq_length
>>>>>>> 257e39a5ad4ce051b1a491b5aa056c1a6cc15889
                        
                    if dn >= 1:    
                        ndx = np.arange(dn*t, dn*(t+1), 1)      
                        if t < dr:
                            ndx = np.append(ndx, np.array(ids+t))
                       
                    else:
<<<<<<< HEAD
                        #ndx = np.array([int(t*dn)])
                        ndx = np.array([int(t*dn_r)]) # corrected
=======
                        ndx = np.array([int(t*dn)])
>>>>>>> 257e39a5ad4ce051b1a491b5aa056c1a6cc15889
                        
                    if self.args.timestep_emb == False:
                        ndx = None
                    #---------------------------------------------

                    if self.args.flag_batch_wise_vqvae_loss:
                        if self.args.recon_type ==1:
                            vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                                self.vae.loss_function_batch(state_input, recon, z_e, latent_emb, ndx=ndx)
                        elif self.args.recon_type ==2:
                            vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                                self.vae.loss_function_batch(Cqt_target, recon, z_e, latent_emb, ndx=ndx)
                        elif self.args.recon_type ==3:
                            vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                                self.vae.loss_function_batch(state_input, recon, z_e, latent_emb, ndx=ndx, Cqt=Cqt_target, recon_Cqt=Cqt_hat)                        
                        vae_losses.append(vae_loss) 
                        ce_losses       += th.mean(ce_loss       )    
                        vq_losses       += th.mean(vq_loss       )
                        commit_losses   += th.mean(commit_loss   )
                        coverage_losses += th.mean(coverage_loss )
                    else:
                        if self.args.recon_type ==1:
                            vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                                self.vae.loss_function(state_input, recon, z_e, latent_emb, ndx=ndx)
                        elif self.args.recon_type ==2:
                            vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                                self.vae.loss_function(Cqt_target, recon, z_e, latent_emb, ndx=ndx)
                        elif self.args.recon_type ==3:
                            vae_loss, ce_loss, vq_loss, commit_loss, coverage_loss = \
                                self.vae.loss_function(state_input, recon, z_e, latent_emb, ndx=ndx, Cqt=Cqt_target, recon_Cqt=Cqt_hat)                        
                        
                        # this results are already computed by taking average in batch-wise
                        vae_losses      += vae_loss 
                        ce_losses       += ce_loss
                        vq_losses       += vq_loss
                        commit_losses   += commit_loss
                        coverage_losses += coverage_loss
                   
        #==============================================================================

        #td = time.time()-ts
        #print( str(td))
        mac_out     = th.stack(mac_out, dim=1)      # Concat over time
        
        if self.args.use_vqvae:            
            visit_nodes = th.stack(visit_nodes, dim=1)  # Concat over time # sequence of trajectory
            
            if self.update_vqvae: 
                if self.args.flag_batch_wise_vqvae_loss:
                    vae_losses      = th.stack(vae_losses, dim=1) # [bs, t]
                    vae_losses      = vae_losses[:,:-1].unsqueeze(-1)
                else:
                    vae_losses      /= batch.max_seq_length # compute average by timestep
                ce_losses       /= batch.max_seq_length
                vq_losses       /= batch.max_seq_length
                commit_losses   /= batch.max_seq_length
                coverage_losses /= batch.max_seq_length
            else: # for monitoring
                #vae_losses        = self.vae_losses     
                vae_losses        = th.tensor(0.0).to(self.args.device)
                ce_losses         = self.ce_losses      
                vq_losses         = self.vq_losses      
                commit_losses     = self.commit_losses  
                coverage_losses   = self.coverage_losses
                
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            
        #.. (c) goal-reaching trajectory generation and (d) intrinsic reward generation-----
        qec_input = chosen_action_qvals.clone().detach()
        eta  = th.zeros_like(qec_input).detach().to(self.args.device)
        #ts = time.time()
        if self.args.goal_reward == True and self.args.use_vqvae == True:
            if self.vae.flag_buffer_update == True:
                for i in range(self.args.batch_size):
                    reference_seq = []
                
                    for t in range(batch.max_seq_length):
                    #.. (c.0) insert updated sequence to codebook keeping top-k 
                        if (t % self.args.trj_update_freq == 0):
                            idx0 = visit_nodes[i][t].clone().item()
                            if self.args.buffer_update_ref == 1: # sum_rewards
                                Cq0 = sum_rewards[i]
                            elif self.args.buffer_update_ref == 2: # reward_to_go
                                Cq0 = reward_tgo[i,t]
                            seq = visit_nodes[i][t:]
                            self.vae.Buffer_seq[idx0].push(Cq0, seq, flag_des_trj[i] ) # insert sequence ( takes time O(log(k)) )                  
                            idx_prev = idx0
                            
                    #.. (c.1) sample a reference trajectory from buffer                                                
                            if self.vae.Buffer_seq[idx0].CanSample():
                                reference_Cq0, reference_seq, reference_des = self.vae.Buffer_seq[idx0].sample_seq()
                        else:
                            idx  = visit_nodes[i][t].item()
                            if (self.args.flag_zero_state_management==False) or (self.args.flag_zero_state_management==True and idx != self.args.n_codes):
                                
                    #.. (c.2) intrinsic reward generation for desirable transition in latent space
                                if self.args.flag_desirability: # only give additional incentive for goal-reaching trajectory
                                    if (idx in reference_seq) and idx != idx_prev and reference_des: 
                                        if self.args.incentive_type == 1:
                                            Cqt = self.vae.call_Cqt(idx)
                                        else: # optimistic view
                                            Cqt = reference_Cq0 # not accurate
                                        eta[i][t-1] = max(Cqt - target_max_qvals[i][t-1], 0.0)
                                else:
                                    if (idx in reference_seq) and idx != idx_prev:
                                        if self.args.incentive_type == 1:
                                            Cqt = self.vae.call_Cqt(idx)
                                        else: # optimistic view
                                            Cqt = reference_Cq0 # not accurate
                                        eta[i][t-1] = max(Cqt - target_max_qvals[i][t-1], 0.0)
                            else:
                                eta[i][t-1] = 0.0 
                            idx_prev = idx
        #------------------------------------------------
        #td = time.time()-ts
        #print( str(td))
        # Calculate 1-step Q-Learning targets
        targets = self.args.gamma * eta * float(self.args.goal_reward)  + rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        
        # Normal L2 loss, take mean over actual data
        #loss = (masked_td_error ** 2).sum() / mask.sum()
        loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.flag_batch_wise_vqvae_loss:
            masked_vae_losses = vae_losses * mask      
            vae_final_loss = (masked_vae_losses).sum()/ mask.sum()
        
        # Optimise
        #.. policy learning
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        #.. VQ-VAE learning
        #ts = time.time()
        if self.args.use_vqvae and self.update_vqvae:            
            self.vae_optimizer.zero_grad()
            grad_norm = th.nn.utils.clip_grad_norm_(self.vae_params, self.args.grad_norm_clip)
            if self.args.flag_batch_wise_vqvae_loss:            
                vae_final_loss.backward()
            else:
                vae_losses.backward()
            
            self.vae_optimizer.step()
        else:
            vae_final_loss = th.tensor(0.0).to(self.args.device)
        #td = time.time()-ts
        #print(str(td))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("eta", eta.mean().item(), t_env)
            if self.args.use_vqvae:
                if self.args.flag_batch_wise_vqvae_loss:
                    self.logger.log_stat("vae_loss", vae_final_loss.item(), t_env)
                else:
                    self.logger.log_stat("vae_loss", vae_losses.item(), t_env)

                self.logger.log_stat("ce_loss", ce_losses.item(), t_env)
                self.logger.log_stat("vq_loss", vq_losses.item(), t_env)
                self.logger.log_stat("commit_loss", commit_losses.item(), t_env)
                self.logger.log_stat("coverage_loss", coverage_losses.item(), t_env)
                
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
        if (t_env - self.vae_log_t >= self.args.learner_log_interval / 100 ) and self.args.use_vqvae:
            if self.args.flag_batch_wise_vqvae_loss:
                self.logger.log_stat("vae_loss", vae_final_loss.item(), t_env)
            else:
                self.logger.log_stat("vae_loss", vae_losses.item(), t_env)
            self.vae_log_t = t_env
            
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.args.use_vqvae:
            self.vae.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)        
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.args.use_vqvae:
            th.save(self.vae.state_dict(), "{}/vae.th".format(path))
            th.save(self.vae.emb.state_dict(), "{}/codebook.th".format(path))
            # additional codebook infomation
            self.vae.save_vae_info(path) 
        
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.args.n_agents

        return input_shape

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.use_vqvae:
            self.vae.load_state_dict(th.load("{}/vae.th".format(path), map_location=lambda storage, loc: storage))
            self.vae.emb.load_state_dict(th.load("{}/codebook.th".format(path), map_location=lambda storage, loc: storage))