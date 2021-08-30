from math import e
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import warnings 
import math
from agent import Agent, PolicyAgent
warnings.filterwarnings("ignore")


def entropy_e(ts):
    '''
    Assuming ts is 2d tensor
    '''
    ent = 0
    for t in ts:
        for p in t:
            ent -= p * T.log(p)
    return ent
    

class MOIRLAgent():
    '''
    The multi-objective Inverse reinforcement learning Agent for conversational search.
    This agent has multiple policies each represented by one <agent> object.
    '''
    def __init__(self, n_policy, n_action, observation_dim, top_n, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_policy = n_policy
        self.n_action = n_action
        self.top_n = top_n
        self.prior = np.exp(np.random.uniform(0, 1, n_policy))
        self.prior /= sum(self.prior)
        self.prior = T.Tensor(self.prior)
        self.policylist = []
        self.loss = nn.NLLLoss()    
        #self.loss = nn.MSELoss()  
        #self.loss = nn.CrossEntropyLoss()    
        self.device = T.device("cuda")
        self.entropy_weight = 1e-3*T.ones(1).to(self.device)

        for i in range(n_policy):
            self.policylist.append(PolicyAgent(n_actions = n_action, input_dims = (3 + self.top_n) * observation_dim + 1 + self.top_n, epsilon = 0))
        self.params = sum([list(policy.Q.parameters()) for policy in self.policylist], [])
        self.params.append(self.entropy_weight)
        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
    
    def save(self, path):
        for i, policy in enumerate(self.policylist):
            T.save(policy.Q.state_dict(), path + str(i))
    
    def load(self, path):
        for i, policy in enumerate(self.policylist):
            policy.Q.load_state_dict(T.load(path + str(i)))


    def E_step(self, conversation_trajectories):
        '''
        The E step of EM algorithm.
        Compute the probability of each conversation derived from each policy.
        '''
        self.z = T.tensor(self.prior.repeat(len(conversation_trajectories),1)).to(self.device)
        for i, traj in enumerate(conversation_trajectories):
            # Each trajectory is a list of (state, action) pairs
            for j, policy in enumerate(self.policylist):
                for s,a in traj:
                    # We need to compute the pi(s,a) using current parameters for each policy function
                    self.z[i,j] *= policy.Q.forward(s)[a]
            # self.z[i] = T.exp(self.z[i])
            self.z[i] /= sum(self.z[i])


    def M_step(self, pos_conversation_trajectories, all_conversation_trajectories):
        '''
        M step of the EM algorithm.
        Update prior distribution of policies and each policy function.
        '''
        self.prior = T.mean(self.z, dim=0) # update prior by averaging over samples ('i's)
        # update policy
        batch_loss = 0
        for j, policy in enumerate(self.policylist):
            self.optimizer.zero_grad()
            L = T.zeros(1).to(self.device)
            
            print("Before update")
            for i, traj in enumerate(pos_conversation_trajectories):
                target_a = T.tensor([a for s,a in traj]).to(self.device)
                # target_a = T.tensor([[1.0, 0.5] if a==0 else [0.0, 1.0]  for s,a in traj]).to(self.device)
                traj_s = T.stack(([s for s,a in traj])).to(self.device)
                predicted_a = policy.Q.forward(traj_s).to(self.device)
                L += self.z[i,j] * self.loss(predicted_a,target_a)
                print(predicted_a, target_a)
                
            for k, traj in enumerate(all_conversation_trajectories):
                all_target_a = T.tensor([1-int(a) for s,a in traj]).to(self.device)
                #all_target_a = T.tensor([[1.0, 0.5] if a==0 else [0.0, 1.0] for s,a in traj]).to(self.device)
                all_traj_s = T.stack(([s for s,a in traj])).to(self.device)
                all_predicted_a = policy.Q.forward(all_traj_s).to(self.device)
                #L += self.z[k,j] * self.loss(all_predicted_a, all_target_a)
                #print(all_predicted_a, all_target_a)
                

                '''
                for k, all_traj in enumerate(all_conversation_trajectories[i]):
                    # all_t, p = all_traj
                    all_t = all_traj
                    # all_target_a = T.tensor([a for s,a in all_t]).to(self.device)
                    all_target_a = T.tensor([[1.0, 0.0] if a==0 else [0.0, 1.0] for s,a in all_t]).to(self.device)
                    all_traj_s = T.stack(([s for s,a in all_t])).to(self.device)
                    all_predicted_a = policy.Q.forward(all_traj_s).to(self.device)
                    #L -= self.z[i,j] * p * self.loss(all_predicted_a, all_target_a)
                    L -= self.z[i,j] * self.loss(all_predicted_a, all_target_a)
                '''
            
            L = L.to(self.device)
            L.backward(retain_graph=True)
            self.optimizer.step()   
            print("Post update")
            for i, traj in enumerate(pos_conversation_trajectories):
                target_a = T.tensor([a for s,a in traj]).to(self.device)
                traj_s = T.stack(([s for s,a in traj])).to(self.device)
                predicted_a = policy.Q.forward(traj_s).to(self.device)
                print(predicted_a, target_a)
            batch_loss += L.detach().item()
        
        #self.scheduler.step()
        return batch_loss
    

    def gail_step(self, self_traj):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        It is still within multi-objective framework.
        '''
        self.prior = T.mean(self.z, dim=0) # update prior by averaging over samples ('i's)
        # update policy
        batch_loss = 0
        for j, policy in enumerate(self.policylist):
            self.optimizer.zero_grad()
            L = T.zeros(1).to(self.device)
            
            for k, traj in enumerate(self_traj):
                s_a_list, p = traj
                a_list = T.tensor([a for s,a,_ in s_a_list]).to(self.device)
                s_list = T.stack(([s for s,a,_ in s_a_list])).to(self.device)
                predicted_a_list = policy.Q.forward(s_list).to(self.device)
                print(predicted_a_list)
                L -= self.z[k,j] * (1-p) * self.loss(predicted_a_list, a_list) - self.entropy_weight * entropy_e(predicted_a_list)
                #print(all_predicted_a, all_target_a)

            L = L.to(self.device)
            L.backward(retain_graph=True)
            self.optimizer.step()   
            batch_loss += L.detach().item()
        
        #self.scheduler.step()
        return batch_loss


    def inference_step(self, cur_traj, state):
        '''
        The inference step of moirl agent first computes the posterior distribution of policies given existing conversation trajectory.
        Then the distribution of policies can be used to compute a weighted reward function.
        '''

        # computing posterior distribution
        post = T.clone(self.prior).to(self.device)
        likelihood = T.ones(self.prior.shape).to(self.device)
        for s,a,_ in cur_traj:
            for j, policy in enumerate(self.policylist):
                likelihood[j] *= T.exp(policy.Q.forward(s).to(self.device)[a])/sum(T.exp(policy.Q.forward(s).to(self.device)))
                
        post *= likelihood
        post /= sum(post)

        weighted_reward = T.zeros(self.n_action).to(self.device)
        for j, policy in enumerate(self.policylist):
            weighted_reward += policy.Q.forward(state) * post[j]
        print(weighted_reward)
        action = T.argmax(weighted_reward).item()

        return state, action
