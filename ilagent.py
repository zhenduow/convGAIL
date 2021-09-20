from math import e
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import warnings 
import math
from T.distributions import Categorical
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


class LinearDeepNetwork(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size = 16):
        super(LinearDeepNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(),
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                T.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.net.apply(init_weights)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)


class ILAgent():
    '''
    The multi-objective Inverse reinforcement learning Agent for conversational search.
    This agent has multiple policies each represented by one <agent> object.
    '''
    def __init__(self, n_action, observation_dim, top_n, lr, lrdc, weight_decay):
        self.lr = lr
        self.lrdc = lrdc
        self.weight_decay = weight_decay
        self.n_action = n_action
        self.top_n = top_n
        self.loss = nn.CrossEntropyLoss()    
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.entropy_weight = 1e-3*T.ones(1).to(self.device)
        self.policy = LinearDeepNetwork(n_actions = n_action, input_dims = (2) * observation_dim + (2) * self.top_n)
        self.params = self.policy.parameters()
        #self.params.append(self.entropy_weight)
        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lrdc)
    
    def save(self, path):
        T.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(T.load(path))

    def gail_step(self, all_expert_traj, all_self_traj):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0
        self.optimizer.zero_grad()
        L = T.zeros(1).to(self.device)


        print("prior update")
        '''  
        for j, traj in enumerate(all_expert_traj):
            expert_a_list = T.tensor([a for s,a in traj]).to(self.device)
            expert_s_list = T.stack(([s for s,a in traj])).to(self.device)
            predicted_a_list = self.policy.forward(expert_s_list).to(self.device)
            L += self.loss(predicted_a_list, expert_a_list)
        '''
        
        for k, traj in enumerate(all_self_traj):
            self_s_a_list, p = traj
            self_a_list = T.tensor([a for s,a,_ in self_s_a_list]).to(self.device)
            self_s_list = T.stack(([s for s,a,_ in self_s_a_list])).to(self.device)
            predicted_a_list = self.policy.forward(self_s_list).to(self.device)
            L +=  -(1-p) * self.loss(predicted_a_list, self_a_list) - self.entropy_weight * entropy_e(predicted_a_list)
            print(predicted_a_list, self_a_list)

        L = L.to(self.device)
        L.backward() 
        self.optimizer.step()
        batch_loss += L.detach().item()

        print("post update")  
        for k, traj in enumerate(all_self_traj):
            self_s_a_list, p = traj
            self_a_list = T.tensor([a for s,a,_ in self_s_a_list]).to(self.device)
            self_s_list = T.stack(([s for s,a,_ in self_s_a_list])).to(self.device)
            predicted_a_list = self.policy.forward(self_s_list).to(self.device)
            print(predicted_a_list, self_a_list)
    
        return batch_loss


    def inference_step(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores):
        '''
        The inference step of moirl agent first computes the posterior distribution of policies given existing conversation trajectory.
        Then the distribution of policies can be used to compute a weighted reward function.
        '''
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        #encoded_state = T.cat((encoded_state, encoded_q), dim=0)
        #encoded_state = T.cat((encoded_state, answers_embeddings[0]), dim=0)
        encoded_state = T.cat((encoded_state, questions_scores[:self.top_n]), dim=0)
        encoded_state = T.cat((encoded_state, answers_scores[:self.top_n]), dim=0)
        state = T.tensor(encoded_state, dtype=T.float).to(self.device)
        pp = self.policy.forward(state)
        action = T.argmax(pp).item()
        #dist = Categorical(action)
        #sample_action = dist.sample().item()
        return state, action
